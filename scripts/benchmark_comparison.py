"""
Benchmark comparison: CLI driver for closed-loop policy evaluation.

Runs one or more robot policies (SLaTe-PRO / hybrid / plain-LLM / oracle / noop)
via ``closed_loop_eval.evaluate_closed_loop``. Script phases (warm-start /
finetune / test) are controlled by ``--warmup_days``, ``--test_days``, and
``--train_every_n_days`` — not by ``--train_days`` alone (that flag only selects
which script files become the eval pool after the train split).

Usage:
  python scripts/benchmark_comparison.py \\
      --slatepro_checkpoint checkpoints/slate_pro \\
      --household A \\
      --train_days 65 \\
      --out_dir results/benchmark
"""

import argparse
import copy
import glob as glob_module
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'SLaTe-PRO'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'STREAK'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'STREAK', 'helpers'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'rail_tasksim'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'rail_tasksim', 'simulation'))

import torch

from simulation import load_config
from closed_loop_eval import _json_sanitize, evaluate_closed_loop


def _discover_scripts(data_dir: str, household: str):
    """Return sorted list of all script paths (train + test) for a household.

    Train scripts come first (sorted), then test scripts (sorted), so that
    ``scripts[:train_days]`` gives training data and ``scripts[train_days:]``
    gives test data.
    """
    train_dir = os.path.join(data_dir, f'Household{household}', 'scripts_train')
    test_dir  = os.path.join(data_dir, f'Household{household}', 'scripts_test')
    train = sorted(glob_module.glob(os.path.join(train_dir, '*.txt'))) if os.path.isdir(train_dir) else []
    test  = sorted(glob_module.glob(os.path.join(test_dir,  '*.txt'))) if os.path.isdir(test_dir)  else []
    return train + test


def _build_slatepro_policy(sim_config: dict):
    """Instantiate ``RobotPolicySlatepro``; enables train-from-scratch if no checkpoint."""
    from agents.robot_policy_slatepro import RobotPolicySlatepro
    if sim_config['robot']['slate_pro']['model_checkpoint'] is None:
        sim_config['robot']['slate_pro']['train_from_scratch'] = True
    return RobotPolicySlatepro(sim_config)


def _build_streak_policy(sim_config: dict, *, cl_mode='none', ewc_lambda=None,
                         lookahead_steps=None, confidence_threshold=None,
                         max_actions=None):
    """Instantiate ``RobotPolicyStreak`` for closed-loop benchmarks (always from scratch)."""
    from agents.robot_policy_streak import RobotPolicyStreak
    sim_config['robot']['streak']['train_from_scratch'] = True
    sim_config['robot']['streak']['model_checkpoint'] = None
    sim_config['robot']['streak']['cl_mode'] = cl_mode
    sim_config['robot']['streak']['ewc_lambda'] = ewc_lambda
    if lookahead_steps is not None:
        sim_config['robot']['streak']['lookahead_steps'] = int(lookahead_steps)
    if confidence_threshold is not None:
        sim_config['robot']['streak']['confidence_threshold'] = float(confidence_threshold)
    if max_actions is not None:
        sim_config['robot']['streak']['max_actions'] = int(max_actions)
    return RobotPolicyStreak(sim_config)


def main():
    """Parse CLI, configure sim/human, run selected policies, write summary JSON."""
    parser = argparse.ArgumentParser(
        description='Benchmark robot policies (SLaTe-PRO / hybrid / plain-LLM / oracle) '
                    'in closed-loop simulation'
    )
    parser.add_argument('--slatepro_checkpoint', default=None,
                        help='SLaTe-PRO checkpoint dir or .ckpt file')
    parser.add_argument('--data_dir', default='external/HOMER_PLUS')
    parser.add_argument('--household', default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--train_days', type=int, default=65,
                        help='Number of scripts used for training (eval on rest)')
    parser.add_argument('--out_dir', default='results/benchmark')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Action confidence threshold (overrides config)')
    parser.add_argument('--skip_slatepro', action='store_true',
                        help='Skip SLaTe-PRO evaluation')
    parser.add_argument('--skip_streak', action='store_true',
                        help='Skip STREAK evaluation (default when --run_streak is not set)')
    parser.add_argument('--run_streak', action='store_true', default=False,
                        help='Run STREAK closed-loop baseline (train-from-scratch by default)')
    parser.add_argument('--streak_lookahead_steps', type=int, default=None,
                        help='Override STREAK lookahead_steps (e.g. 1 for 1-step paper parity)')
    parser.add_argument('--streak_max_actions', type=int, default=None,
                        help='Cap STREAK to the top-N valid relocations by '
                        'confidence per tick (None = uncapped paper default)')
    parser.add_argument('--streak_cl_mode', default='none', choices=['none', 'l2', 'ewc'],
                        help='STREAK continual-learning mode; each train() round is one task')
    parser.add_argument('--streak_ewc_lambda', type=float, default=None,
                        help='CL regularization strength (default: l2=200, ewc=100)')
    parser.add_argument('--skip_noop', action='store_true',
                        help='Skip noop baseline evaluation')
    parser.add_argument('--time_step_minutes', type=int, default=None,
                        help='Override simulation timestep (minutes). '
                             'Use 10 for SLaTe-PRO trained on HOMER+ dt=10.')
    parser.add_argument('--lookahead_steps', type=str, default=None,
                        help='Override SLaTe-PRO lookahead model')
    parser.add_argument('--lookahead_eval', type=str, default=None,
                        help='Override SLaTe-PRO lookahead eval')
    parser.add_argument('--suppress_robot_action', action='store_true',
                        help='Give robot clean (no-robot-action) world state '
                             'to isolate feedback loop effect')
    parser.add_argument('--active_start_minutes', type=int, default=360,
                        help='Predict-only: first timestep to evaluate (default 360). '
                             'Set to 360 (6 AM) to match batch .pt data range.')
    parser.add_argument('--active_end_minutes', type=int, default=1440,
                        help='Predict-only: last timestep to evaluate (default 1440).')
    parser.add_argument('--all_scripts', action='store_true',
                        help='Evaluate all scripts, not just the test scripts')
    parser.add_argument('--warmup_days', type=int, default=5,
                        help='Number of days to warmup the model before evaluation')
    parser.add_argument('--test_days', type=int, default=10,
                        help='Number of days to test the model')
    parser.add_argument('--train_every_n_days', type=int, default=1,
                        help='Number of days to train the model every n days')
    parser.add_argument('--no_reset_env', action='store_true', default=False,
                        help='Do not reset environment between days (world state persists)')
    parser.add_argument('--use_prior_checkpoints', action='store_true', default=False,
                        help='Skip warm-start and training; load existing weights_day*.pt from the prior '
                             'run. Outputs are written to the usual filename; if it already exists, it is '
                             'renamed to *_before_DDMM_HHMM.json first.')
    parser.add_argument('--vigilance_enabled', action='store_true', default=False,
                        help='Enable human vigilance: human observes robot actions and '
                             'corrects unmatched ones (only during finetune/test)')
    parser.add_argument('--vigilance_probability', type=float, default=1.0,
                        help='P(human observes a given robot action) [0,1]')
    parser.add_argument('--vigilance_window_hours', type=float, default=6.0,
                        help='Lookahead window for matching planned human actions (hours)')
    parser.add_argument('--human_policy', type=str, default='scripted',
                        choices=['scripted', 'llm'],
                        help='Human agent policy ("llm" forces vigilance off)')
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini',
                        help='OpenAI model id used by the LLM human agent')
    parser.add_argument('--llm_cache_dir', type=str, default='cache/llm_human/',
                        help='Disk cache directory for LLM decisions')
    parser.add_argument('--llm_max_history_actions', type=int, default=20,
                        help='Per-prompt cap on the day-history events stitched in')
    parser.add_argument('--llm_few_shot_path', type=str,
                        default='config/llm_human_examples.json',
                        help='JSON file with few-shot examples')
    parser.add_argument('--llm_max_calls_per_day', type=int, default=5000,
                        help='Hard backstop cap on LLM calls per simulated day '
                             '(use --llm_max_usd_per_day for cost-based cap)')
    parser.add_argument('--llm_max_usd_per_day', type=float, default=1.0,
                        help='Estimated USD budget per simulated day for LLM calls')
    parser.add_argument('--llm_observe_mode', type=str, default='redirect',
                        choices=['undo_allow', 'redirect'],
                        help='LLM observe-hook mode: redirect (v2, the canonical/only '
                             'human we report) or undo_allow (v1, deprecated/superseded). '
                             'Defaults to redirect so no run silently uses undo_allow.')
    parser.add_argument('--run_hybrid', action='store_true', default=False,
                        help='Evaluate the hybrid LLM+transformer robot policy')
    parser.add_argument('--hybrid_memory_dir', type=str, default=None,
                        help='Curated-memory dir (default: <out_dir>/hybrid_memory_<HH>)')
    parser.add_argument('--hybrid_llm_cache_dir', type=str, default=None,
                        help='Disk cache dir for the hybrid robot LLM calls')
    parser.add_argument('--hybrid_mode', type=str, default=None,
                        choices=['event', 'interval'],
                        help='Hybrid activity-transformer input mode')
    parser.add_argument('--hybrid_curate_model', type=str, default=None,
                        help='LLM for training-time memory curation')
    parser.add_argument('--hybrid_max_usd', type=float, default=None,
                        help='Hard USD ceiling for the hybrid robot LLM clients')
    parser.add_argument('--hybrid_prob_threshold', type=float, default=None,
                        help='Min candidate score to act (hybrid policy)')
    parser.add_argument('--hybrid_max_actions_per_tick', type=int, default=None,
                        help='Max robot actions per tick (hybrid policy)')
    parser.add_argument('--hybrid_pf_p_bg', type=float, default=None,
                        help='Override the PF background rate p_bg with a fixed value (ablation)')
    parser.add_argument('--hybrid_pf_robot_opens', action='store_true',
                        help='PF: robot placements drive full particle state update '
                             '(can open/continue activities); weights untouched')
    parser.add_argument('--hybrid_accept_gate', action='store_true',
                        help='enable the acceptance-learning gate (within-day exp '
                             'backoff + cross-day gamma-decayed P_accept score multiplier)')
    parser.add_argument('--hybrid_prep_discount', type=float, default=None,
                        help='Score discount on next-activity prep actions')
    parser.add_argument('--hybrid_next_act_conf', type=float, default=None,
                        help='Min transformer confidence to prep next activity')
    parser.add_argument('--hybrid_act_window_min', type=float, default=None,
                        help='Only emit actions whose expected moment is '
                             'within this many minutes of now')
    parser.add_argument('--hybrid_no_transformer_recency', action='store_true',
                        default=False, help='CL ablation: disable transformer '
                        'recency weighting (graph decay unaffected)')
    parser.add_argument('--hybrid_no_lead_gate', action='store_true',
                        default=False, help='Ablation: disable the lead-time gate '
                        '(no too-early suppression; candidates emit regardless of '
                        'how far ahead of the human-expected time they are)')
    parser.add_argument('--hybrid_no_cl_forget', action='store_true',
                        default=False, help='CL ablation: pin the prequential '
                        'gamma forgetting factor to 1.0 (no recency discount on '
                        'accumulated evidence); CL still retrains each cycle')
    parser.add_argument('--hybrid_no_graph_recency', action='store_true',
                        default=False, help='CL ablation: still optimize gamma '
                        'but do NOT apply it to the task graphs (uniform day '
                        'weights); gamma is still available to the transformer. '
                        'Combine with --hybrid_no_transformer_recency for '
                        'recency on neither.')
    parser.add_argument('--hybrid_no_precedence_gate', action='store_true',
                        default=False, help='Ablation: disable the precedence '
                        'ordering gate (_gate_precedence); candidates are not '
                        'suppressed for violating learned object orderings')
    parser.add_argument('--hybrid_no_prep', action='store_true',
                        default=False, help='Ablation: disable the prep branch '
                        '(no proactive pre-staging of upcoming activities; PF '
                        'recognition of currently-open activities only). The '
                        'activity transformer is this branch\'s only consumer, so '
                        'this is also the effective "no transformer" ablation.')
    parser.add_argument('--hybrid_no_current', action='store_true',
                        default=False, help='Ablation (only-prep): disable the '
                        'current-activity branch; the planner acts solely on '
                        'transformer-anticipated upcoming activities.')
    parser.add_argument('--hybrid_pf_map', action='store_true',
                        default=False, help='Ablation: collapse the PF open-'
                        'activity posterior to its MAP (single most-likely-open '
                        'activity at weight 1.0; no posterior averaging).')
    parser.add_argument('--hybrid_pf_single_belief', action='store_true',
                        default=False, help='Ablation: remove the PF entirely -- '
                        '1 deterministic particle (argmax association, threshold-0.5 '
                        'open/close). A single greedy belief, no ensemble/recovery.')
    parser.add_argument('--hybrid_accept_buildon', action='store_true',
                        default=False, help='Gate v2: credit an accept only when '
                        'the human builds upon a placement (continued); a stuck/'
                        'untouched placement (active) is neutral, not an accept. '
                        'Implies --hybrid_accept_gate.')
    parser.add_argument('--hybrid_cl', action='store_true', default=False,
                        help='v2-PF continual learning: prequential gamma '
                             'decay on graphs, actor-split counts, recency-'
                             'weighted warm-start transformer')
    parser.add_argument('--llm_redirect_window_min', type=int, default=None,
                        help='LLM-human v2: plan window shown to the LLM and '
                             'searched for redirect targets (minutes)')
    parser.add_argument('--run_oracle', action='store_true', default=False,
                        help='P0 GT-script oracle ceiling policy')
    parser.add_argument('--run_plainllm', action='store_true', default=False,
                        help='Evaluate the plain-LLM baseline robot policy')
    parser.add_argument('--plainllm_model', type=str, default=None)
    parser.add_argument('--plainllm_history_days', type=int, default=None)
    parser.add_argument('--plainllm_cache_dir', type=str, default=None)
    # --- NOPA (Puig et al., ICRA 2023) ---
    parser.add_argument('--run_nopa', action='store_true', default=False,
                        help='Evaluate the NOPA baseline robot policy')
    parser.add_argument('--nopa_solo_days', type=int, default=None,
                        help='Days the goal proposal network trains on. MUST be '
                             'the main-agent-alone (warm-start) days; normally '
                             '= --warmup_days')
    parser.add_argument('--nopa_k', type=int, default=None,
                        help='K goal proposals per resample (paper: 20)')
    parser.add_argument('--nopa_tprop', type=int, default=None,
                        help='Prediction horizon T_prop in plan steps (paper: 15)')
    parser.add_argument('--nopa_lmax', type=float, default=None,
                        help='L_M for goal edges beyond the horizon (paper: 100)')
    parser.add_argument('--nopa_wr', type=float, default=None)
    parser.add_argument('--nopa_wc', type=float, default=None,
                        help='Inert in this domain: L_H == 1 for every edge')
    parser.add_argument('--nopa_wm', type=float, default=None,
                        help='State-disturbance weight (paper w_d: 5)')
    parser.add_argument('--nopa_max_actions', type=int, default=None,
                        help='Subgoals executed per tick (paper Alg. 3: 1)')
    parser.add_argument('--nopa_min_value', type=float, default=None,
                        help='Stay idle unless V exceeds this')
    parser.add_argument('--nopa_lm_units', type=str, default=None,
                        choices=['ticks', 'steps'], help='Unit of L_M(e)')
    parser.add_argument('--nopa_no_horizon_fallback', action='store_true', default=False,
                        help='use each predicate\'s true plan position instead of L_max '
                             'when it falls past T_prop')
    parser.add_argument('--nopa_action_rule', type=str, default=None,
                        choices=['value', 'most_likely'],
                        help="'most_likely' drops Eq.(1) and acts on argmax p(e)")
    parser.add_argument('--nopa_gpn_loss', type=str, default=None,
                        choices=['ce', 'weighted_ce', 'focal'],
                        help='GPN count loss; task-level goals need rebalancing')
    parser.add_argument('--nopa_task_split', type=str, default=None,
                        choices=['none', 'prep_cleanup'],
                        help='NOPA: also split each GT activity into prep/cleanup at the '
                             "human script's second relocation of an object")
    parser.add_argument('--nopa_day_episodes', action='store_true', default=False,
                        help='revert NOPA to one-goal-per-DAY episodes (the original '
                             'port); default is one episode per GT activity task')
    parser.add_argument('--nopa_goal_source', type=str, default=None,
                        choices=['gpn', 'uniform', 'oracle'],
                        help="'uniform' reproduces the paper's OursRG ablation; "
                             "'oracle' reads the GT day goal (measurement instrument)")
    parser.add_argument('--nopa_no_inverse_planning', action='store_true',
                        default=False, help="Paper's Ours-InvPlan ablation")
    parser.add_argument('--nopa_no_restore', action='store_true', default=False,
                        help="Paper's Ours-Return ablation (also pins w_m=0)")
    parser.add_argument('--nopa_restore_all_initial', action='store_true',
                        default=False,
                        help='Literal Alg-3 reading: restore ANY object off its '
                             'initial location, including ones only the human moved')
    parser.add_argument('--nopa_no_plan_order_gate', action='store_true',
                        default=False,
                        help='Literal Alg-3 reading: score every goal edge, '
                             'including several destinations for one object')
    parser.add_argument('--nopa_argmax_goal', action='store_true', default=False,
                        help="Paper's HPGPN baseline (K=1, argmax goal)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Discover scripts and split ────────────────────────────────────────
    all_scripts = _discover_scripts(args.data_dir, args.household)
    print(f'Household {args.household}: {len(all_scripts)} total scripts')
    if args.all_scripts:
        test_scripts = all_scripts
        print(f'  Test on all: {len(test_scripts)}')
    else:
        test_scripts = all_scripts[args.train_days:]
        print(f'  Train: {args.train_days}, Test: {len(test_scripts)}')

    if not test_scripts:
        print('ERROR: No test scripts available')
        return

    # ── Base sim config ───────────────────────────────────────────────────
    sim_config = load_config()
    if args.time_step_minutes is not None:
        sim_config['simulation']['time_step_minutes'] = args.time_step_minutes
    sim_config['simulation']['verbose'] = False
    sim_config['benchmark']['household'] = args.household
    if args.threshold is not None:
        sim_config['robot']['slate_pro']['confidence_threshold'] = args.threshold

    print(f'  Threshold: {args.threshold} -> config: {sim_config["robot"]["slate_pro"]["confidence_threshold"]}')
    if args.lookahead_steps is not None:
        sim_config['robot']['slate_pro']['lookahead_steps'] = int(args.lookahead_steps)
    print(f"  Lookahead model: {sim_config['robot']['slate_pro']['lookahead_steps']}")
    if args.lookahead_eval is not None:
        sim_config['simulation']['lookahead_eval'] = int(args.lookahead_eval)
    print(f"  Lookahead eval: {sim_config['simulation']['lookahead_eval']}")
    print(f'  Clean robot obs: {args.suppress_robot_action}')
    if args.vigilance_enabled:
        sim_config['human'].setdefault('curiosity_vigilance', {})
        sim_config['human']['curiosity_vigilance']['enabled'] = True
        sim_config['human']['curiosity_vigilance']['probability'] = args.vigilance_probability
        sim_config['human']['curiosity_vigilance']['correction_window_hours'] = args.vigilance_window_hours
    print(f'  Vigilance: {args.vigilance_enabled} (p={args.vigilance_probability}, window={args.vigilance_window_hours}h)')

    # --- Human policy (scripted | llm) ---
    sim_config['human']['policy'] = args.human_policy
    if args.human_policy == 'llm':
        sim_config['human'].setdefault('llm', {})
        sim_config['human']['llm'].update({
            'provider':              'openai',
            'model':                 args.llm_model,
            'cache_dir':             args.llm_cache_dir,
            'max_history_actions':   args.llm_max_history_actions,
            'few_shot_path':         args.llm_few_shot_path,
            'max_llm_calls_per_day': args.llm_max_calls_per_day,
            'max_usd_per_day':       args.llm_max_usd_per_day,
            'observe_mode':          args.llm_observe_mode,
        })
        if args.llm_redirect_window_min is not None:
            sim_config['human']['llm']['redirect_window_min'] = args.llm_redirect_window_min
        # Force legacy vigilance off when LLM drives the decisions.
        sim_config['human'].setdefault('curiosity_vigilance', {})['enabled'] = False
        print(f"  Human policy: llm  model={args.llm_model}  "
              f"cache={args.llm_cache_dir}  observe_mode={args.llm_observe_mode}  "
              f"max_calls/day={args.llm_max_calls_per_day}")
    else:
        print('  Human policy: scripted')

    results = {}

    script_distribution = [args.warmup_days, -1, args.test_days]
    print(f'  Script distribution: {script_distribution}')

    # ── SLaTe-PRO evaluation ─────────────────────────────────────────────
    if not args.skip_slatepro:
        print(f'\n=== SLaTe-PRO (closed_loop) ===')
        sp_config = copy.deepcopy(sim_config)
        sp_config['robot']['policy'] = 'slate_pro'
        sp_config['robot']['slate_pro']['model_checkpoint'] = args.slatepro_checkpoint
        sp_config['household'] = args.household

        try:
            sp_policy = _build_slatepro_policy(sp_config)
            sp_results = evaluate_closed_loop(
                sp_policy, sp_config, test_scripts,
                suppress_robot_action=args.suppress_robot_action,
                script_distribution=script_distribution,
                active_start_minutes=args.active_start_minutes,
                active_end_minutes=args.active_end_minutes,
                train_every_n_days=args.train_every_n_days,
                reset_env=not args.no_reset_env,
                out_path = os.path.join(args.out_dir, f'benchmark_slatepro_{args.household}.json'),
                use_prior_checkpoints=args.use_prior_checkpoints)
            results['slatepro'] = sp_results
            # Print summary from the last non-empty phase
            for _phase in ['test', 'finetune', 'warm-start']:
                t = sp_results['totals'].get(_phase, {})
                if t:
                    hm = t.get("human_metrics", {}) or {}
                    em = t.get("env_metrics", {}) or {}
                    print(f'  [{_phase}] Skipped={hm.get("skipped", 0)}  Corrected={hm.get("corrected", 0)}  '
                          f'DoneAsIs={hm.get("done_as_is", 0)}')
                    print(f'  [{_phase}] Precision={t.get("human_prec", 0.0):.4f}  Recall={t.get("human_rec", 0.0):.4f}  '
                          f'F1={t.get("human_f1", 0.0):.4f}')
                    print(f'  [{_phase}] TP={em.get("precision_correct", 0)}  '
                          f'WrongDest={em.get("precision_wrong_destination", 0)}  '
                          f'WrongObj={em.get("precision_wrong_object", 0)}  '
                          f'Missed={em.get("moved_missed", 0)}')
        except Exception as e:
            import traceback
            print(f'  SLaTe-PRO eval failed: {e}')
            traceback.print_exc()
            results['slatepro'] = None
            raise e

    # ── STREAK evaluation ───────────────────────────────────────────────
    if args.run_streak and not args.skip_streak:
        print(f'\n=== STREAK (closed_loop, train-from-scratch) ===')
        st_config = copy.deepcopy(sim_config)
        st_config['robot']['policy'] = 'streak'
        st_config['household'] = args.household

        try:
            st_policy = _build_streak_policy(
                st_config, cl_mode=args.streak_cl_mode, ewc_lambda=args.streak_ewc_lambda,
                lookahead_steps=args.streak_lookahead_steps,
                confidence_threshold=args.threshold,
                max_actions=args.streak_max_actions)
            st_results = evaluate_closed_loop(
                st_policy, st_config, test_scripts,
                suppress_robot_action=args.suppress_robot_action,
                script_distribution=script_distribution,
                active_start_minutes=args.active_start_minutes,
                active_end_minutes=args.active_end_minutes,
                train_every_n_days=args.train_every_n_days,
                reset_env=not args.no_reset_env,
                out_path=os.path.join(args.out_dir, f'benchmark_streak_{args.household}.json'),
                use_prior_checkpoints=args.use_prior_checkpoints)
            results['streak'] = st_results
            for _phase in ['test', 'finetune', 'warm-start']:
                t = st_results['totals'].get(_phase, {})
                if t:
                    hm = t.get("human_metrics", {}) or {}
                    em = t.get("env_metrics", {}) or {}
                    print(f'  [{_phase}] Skipped={hm.get("skipped", 0)}  Corrected={hm.get("corrected", 0)}  '
                          f'DoneAsIs={hm.get("done_as_is", 0)}')
                    print(f'  [{_phase}] Precision={t.get("human_prec", 0.0):.4f}  Recall={t.get("human_rec", 0.0):.4f}  '
                          f'F1={t.get("human_f1", 0.0):.4f}')
                    print(f'  [{_phase}] TP={em.get("precision_correct", 0)}  '
                          f'WrongDest={em.get("precision_wrong_destination", 0)}  '
                          f'WrongObj={em.get("precision_wrong_object", 0)}  '
                          f'Missed={em.get("moved_missed", 0)}')
        except Exception as e:
            import traceback
            print(f'  STREAK eval failed: {e}')
            traceback.print_exc()
            results['streak'] = None
            raise e

    # ── Hybrid LLM+transformer evaluation ────────────────────────────────
    if args.run_hybrid:
        print(f'\n=== Hybrid LLM+Transformer (closed_loop) ===')
        hy_config = copy.deepcopy(sim_config)
        hy_config['robot']['policy'] = 'hybrid'
        hy_cfg = hy_config['robot']['hybrid']
        hy_cfg['memory_dir'] = (args.hybrid_memory_dir or
                                os.path.join(args.out_dir,
                                             f'hybrid_memory_{args.household}'))
        if args.hybrid_llm_cache_dir is not None:
            hy_cfg['llm']['cache_dir'] = args.hybrid_llm_cache_dir
        if args.hybrid_mode is not None:
            hy_cfg['transformer']['mode'] = args.hybrid_mode
        if args.hybrid_curate_model is not None:
            hy_cfg['llm']['curate_model'] = args.hybrid_curate_model
        if args.hybrid_max_usd is not None:
            hy_cfg['llm']['max_usd'] = args.hybrid_max_usd
        if args.hybrid_prob_threshold is not None:
            hy_cfg['policy']['prob_threshold'] = args.hybrid_prob_threshold
        if args.hybrid_pf_p_bg is not None:
            hy_cfg['policy']['pf_p_bg'] = args.hybrid_pf_p_bg
        if args.hybrid_pf_robot_opens:
            hy_cfg['policy']['pf_robot_opens'] = True
        if args.hybrid_max_actions_per_tick is not None:
            hy_cfg['policy']['max_actions_per_tick'] = args.hybrid_max_actions_per_tick
        if args.hybrid_accept_gate:
            hy_cfg['policy']['accept_gate'] = True
        if args.hybrid_accept_buildon:
            # build-upon (v2) operates on the gate's accept accounting, so it
            # implies the gate is active.
            hy_cfg['policy']['accept_buildon'] = True
            hy_cfg['policy']['accept_gate'] = True
        if args.hybrid_prep_discount is not None:
            hy_cfg['policy']['prep_discount'] = args.hybrid_prep_discount
        if args.hybrid_next_act_conf is not None:
            hy_cfg['policy']['next_act_conf'] = args.hybrid_next_act_conf
        if args.hybrid_act_window_min is not None:
            hy_cfg['policy']['act_window_min'] = args.hybrid_act_window_min
        if args.hybrid_cl:
            hy_cfg.setdefault('continual_learning', {})['enabled'] = True
        if args.hybrid_no_transformer_recency:
            hy_cfg.setdefault('continual_learning', {})['transformer_recency'] = False
        if args.hybrid_no_lead_gate:
            hy_cfg['policy']['lead_time_gate'] = False
        if args.hybrid_no_cl_forget:
            hy_cfg.setdefault('continual_learning', {})['cl_forget'] = False
        if args.hybrid_no_graph_recency:
            hy_cfg.setdefault('continual_learning', {})['graph_recency'] = False
        if args.hybrid_no_precedence_gate:
            hy_cfg['policy']['precedence_gate'] = False
        if args.hybrid_no_prep:
            hy_cfg['policy']['prep_branch'] = False
        if args.hybrid_no_current:
            hy_cfg['policy']['current_branch'] = False
        if args.hybrid_pf_map:
            hy_cfg['policy']['pf_map'] = True
        if args.hybrid_pf_single_belief:
            hy_cfg['policy']['pf_single_belief'] = True
        print(f"  Hybrid policy: threshold={hy_cfg['policy']['prob_threshold']} "
              f"cap={hy_cfg['policy']['max_actions_per_tick']} "
              f"prep_discount={hy_cfg['policy']['prep_discount']} "
              f"next_act_conf={hy_cfg['policy']['next_act_conf']}")
        print(f"  Hybrid: mode={hy_cfg['transformer']['mode']}  "
              f"memory={hy_cfg['memory_dir']}  "
              f"curate={hy_cfg['llm']['curate_model']}  "
              f"cache={hy_cfg['llm']['cache_dir']}")

        from agents.robot_policy_hybrid import RobotPolicyHybrid
        hy_policy = RobotPolicyHybrid(hy_config)
        hy_results = evaluate_closed_loop(
            hy_policy, hy_config, test_scripts,
            suppress_robot_action=args.suppress_robot_action,
            script_distribution=script_distribution,
            active_start_minutes=args.active_start_minutes,
            active_end_minutes=args.active_end_minutes,
            train_every_n_days=args.train_every_n_days,
            reset_env=not args.no_reset_env,
            out_path=os.path.join(args.out_dir, f'benchmark_hybrid_{args.household}.json'),
            use_prior_checkpoints=args.use_prior_checkpoints)
        results['hybrid'] = hy_results
        print(f"  Hybrid LLM stats: {hy_policy.curator.stats()}")
        for _phase in ['test', 'finetune', 'warm-start']:
            t = hy_results['totals'].get(_phase, {})
            if t:
                hm = t.get("human_metrics", {}) or {}
                print(f'  [{_phase}] Skipped={hm.get("skipped", 0)}  Corrected={hm.get("corrected", 0)}  '
                      f'DoneAsIs={hm.get("done_as_is", 0)}')
                print(f'  [{_phase}] Precision={t.get("human_prec", 0.0):.4f}  Recall={t.get("human_rec", 0.0):.4f}  '
                      f'F1={t.get("human_f1", 0.0):.4f}')

    # ── Plain-LLM baseline evaluation ────────────────────────────────────
    if args.run_oracle:
        print(f'\n=== Oracle ceiling (closed_loop) ===')
        or_config = copy.deepcopy(sim_config)
        or_config['robot']['policy'] = 'oracle'
        print("  Oracle: pure ceiling, emit=1-tick-ahead (GT from human script_path)")
        from agents.robot_policy_oracle import RobotPolicyOracle
        or_policy = RobotPolicyOracle(or_config)
        or_results = evaluate_closed_loop(
            or_policy, or_config, test_scripts,
            suppress_robot_action=args.suppress_robot_action,
            script_distribution=script_distribution,
            active_start_minutes=args.active_start_minutes,
            active_end_minutes=args.active_end_minutes,
            train_every_n_days=args.train_every_n_days,
            reset_env=not args.no_reset_env,
            out_path=os.path.join(args.out_dir, f'benchmark_oracle_{args.household}.json'),
            use_prior_checkpoints=args.use_prior_checkpoints)
        results['oracle'] = or_results

    if args.run_plainllm:
        print(f'\n=== Plain-LLM baseline (closed_loop) ===')
        pl_config = copy.deepcopy(sim_config)
        pl_config['robot']['policy'] = 'plain_llm'
        pl_cfg = pl_config['robot']['plain_llm']
        if args.plainllm_model is not None:
            pl_cfg['model'] = args.plainllm_model
        if args.plainllm_history_days is not None:
            pl_cfg['history_days'] = args.plainllm_history_days
        if args.plainllm_cache_dir is not None:
            pl_cfg['cache_dir'] = args.plainllm_cache_dir
        print(f"  Plain-LLM: model={pl_cfg['model']} "
              f"history_days={pl_cfg['history_days']} "
              f"max_predictions={pl_cfg['max_predictions']} "
              f"cache={pl_cfg['cache_dir']}")
        from agents.robot_policy_plainllm import RobotPolicyPlainLLM
        pl_policy = RobotPolicyPlainLLM(pl_config)
        pl_results = evaluate_closed_loop(
            pl_policy, pl_config, test_scripts,
            suppress_robot_action=args.suppress_robot_action,
            script_distribution=script_distribution,
            active_start_minutes=args.active_start_minutes,
            active_end_minutes=args.active_end_minutes,
            train_every_n_days=args.train_every_n_days,
            reset_env=not args.no_reset_env,
            out_path=os.path.join(args.out_dir, f'benchmark_plainllm_{args.household}.json'),
            use_prior_checkpoints=args.use_prior_checkpoints)
        results['plainllm'] = pl_results
        print(f"  Plain-LLM stats: {pl_policy.client.stats()}")
        for _phase in ['test', 'finetune', 'warm-start']:
            t = pl_results['totals'].get(_phase, {})
            if t:
                hm = t.get("human_metrics", {}) or {}
                print(f'  [{_phase}] Skipped={hm.get("skipped", 0)}  Corrected={hm.get("corrected", 0)}  '
                      f'DoneAsIs={hm.get("done_as_is", 0)}')
                print(f'  [{_phase}] F1={t.get("human_f1", 0.0):.4f}')

    if args.run_nopa:
        print(f'\n=== NOPA baseline (closed_loop) ===')
        np_config = copy.deepcopy(sim_config)
        np_config['robot']['policy'] = 'nopa'
        np_cfg = np_config['robot']['nopa']
        np_pol = np_cfg['policy']
        np_cfg['solo_days'] = (args.nopa_solo_days if args.nopa_solo_days
                               is not None else args.warmup_days)
        if args.nopa_no_horizon_fallback:
            np_pol['lm_horizon_fallback'] = False
        if args.nopa_action_rule is not None:
            np_pol['action_rule'] = args.nopa_action_rule
        if args.nopa_gpn_loss is not None:
            np_cfg.setdefault('gpn', {})['loss'] = args.nopa_gpn_loss
        if args.nopa_day_episodes:
            np_cfg['task_episodes'] = False
        if args.nopa_task_split is not None:
            np_cfg['task_split'] = args.nopa_task_split
            np_pol['lm_units'] = 'ticks'
        for arg_val, key in ((args.nopa_k, 'K'), (args.nopa_tprop, 'T_prop'),
                             (args.nopa_lmax, 'L_max'), (args.nopa_wr, 'w_r'),
                             (args.nopa_wc, 'w_c'), (args.nopa_wm, 'w_m'),
                             (args.nopa_max_actions, 'max_actions'),
                             (args.nopa_min_value, 'min_value'),
                             (args.nopa_lm_units, 'lm_units'),
                             (args.nopa_goal_source, 'goal_source')):
            if arg_val is not None:
                np_pol[key] = arg_val
        if args.nopa_no_inverse_planning:
            np_pol['inverse_planning'] = False
        if args.nopa_no_restore:
            np_pol['restore'] = False
        if args.nopa_restore_all_initial:
            np_pol['restore_all_initial'] = True
        if args.nopa_no_plan_order_gate:
            np_pol['plan_order_gate'] = False
        if args.nopa_argmax_goal:
            np_pol['argmax_goal'] = True
        print(f"  NOPA: solo_days={np_cfg['solo_days']} K={np_pol['K']} "
              f"T_prop={np_pol['T_prop']} w_m={np_pol['w_m']} "
              f"max_actions={np_pol['max_actions']} "
              f"goal_source={np_pol['goal_source']} "
              f"inv_planning={np_pol['inverse_planning']} "
              f"restore={np_pol['restore']} "
              f"plan_order_gate={np_pol['plan_order_gate']} "
              f"argmax_goal={np_pol['argmax_goal']}")
        from agents.robot_policy_nopa import RobotPolicyNOPA
        np_policy = RobotPolicyNOPA(np_config)
        np_results = evaluate_closed_loop(
            np_policy, np_config, test_scripts,
            suppress_robot_action=args.suppress_robot_action,
            script_distribution=script_distribution,
            active_start_minutes=args.active_start_minutes,
            active_end_minutes=args.active_end_minutes,
            train_every_n_days=args.train_every_n_days,
            reset_env=not args.no_reset_env,
            out_path=os.path.join(args.out_dir, f'benchmark_nopa_{args.household}.json'),
            use_prior_checkpoints=args.use_prior_checkpoints)
        results['nopa'] = np_results
        for _phase in ['test', 'finetune', 'warm-start']:
            t = np_results['totals'].get(_phase, {})
            if t:
                hm = t.get("human_metrics", {}) or {}
                print(f'  [{_phase}] Skipped={hm.get("skipped", 0)}  Corrected={hm.get("corrected", 0)}  '
                      f'DoneAsIs={hm.get("done_as_is", 0)}')
                print(f'  [{_phase}] F1={t.get("human_f1", 0.0):.4f}')

    # ── Noop baseline (no robot) ─────────────────────────────────────────
    if not args.skip_noop:
        print('\n=== Noop Baseline (no robot) ===')
        from agents.robot_policy import RobotPolicyNoop
        noop_config = copy.deepcopy(sim_config)
        noop_config['robot']['policy'] = 'noop'
        noop_policy = RobotPolicyNoop(noop_config)
        noop_results = evaluate_closed_loop(
            noop_policy, noop_config, test_scripts,
            suppress_robot_action=args.suppress_robot_action,
            script_distribution=script_distribution,
            active_start_minutes=args.active_start_minutes,
            active_end_minutes=args.active_end_minutes,
            train_every_n_days=args.train_every_n_days,
            reset_env=not args.no_reset_env,
            use_prior_checkpoints=args.use_prior_checkpoints)
        results['noop'] = noop_results
        for _phase in ['test', 'finetune', 'warm-start']:
            t = noop_results['totals'].get(_phase, {})
            if t:
                hm = t.get("human_metrics", {}) or {}
                print(f'  [{_phase}] Skipped={hm.get("skipped", 0)}  Corrected={hm.get("corrected", 0)}  '
                      f'DoneAsIs={hm.get("done_as_is", 0)}')

    # ── Save results ─────────────────────────────────────────────────────
    out_path = os.path.join(args.out_dir, f'benchmark_{args.household}.json')
    with open(out_path, 'w') as f:
        json.dump(_json_sanitize({
            'household': args.household,
            'train_days': args.train_days,
            'suppress_robot_action': args.suppress_robot_action,
            'vigilance_enabled': args.vigilance_enabled,
            'vigilance_probability': args.vigilance_probability,
            'vigilance_window_hours': args.vigilance_window_hours,
            'reset_env': not args.no_reset_env,
            'test_scripts': [os.path.basename(s) for s in test_scripts],
            'results': results,
        }), f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
