import sys
sys.path.append('external/SLaTe-PRO')
sys.path.append('external/rail_tasksim')

import argparse
import copy
import os
import yaml

from environment.world import World, print_graph_difference
from environment.human_agent import HumanAgent
from agents.robot_policy import RobotMoveAction
from visualization.logger import SimulationLogger
from constants import SIM_START_TIME_MINUTES


# ======================================================================
# Config loading & CLI override
# ======================================================================

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'base.yaml')


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load YAML config file and return as a plain dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _set_nested(d: dict, dotted_key: str, value):
    """Set a value in a nested dict using a dotted key like 'robot.policy'."""
    keys = dotted_key.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _cast_value(value_str: str):
    """Try to cast a CLI string value to the appropriate Python type."""
    if value_str.lower() == 'null' or value_str.lower() == 'none':
        return None
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def validate_config(config: dict) -> None:
    """Validate that *config* contains all required keys with plausible values.

    Raises ``KeyError`` for missing required keys and ``ValueError`` for
    obviously wrong values.  Called after loading YAML + applying CLI overrides
    so that config typos are caught at startup rather than silently using
    hard-coded defaults deep in the code.
    """
    # --- Required top-level sections ---
    for section in ('paths', 'simulation', 'robot'):
        if section not in config:
            raise KeyError(f"Config is missing required section '{section}'")

    paths = config['paths']
    sim   = config['simulation']
    robot = config['robot']

    # --- Paths ---
    if 'init_graph_file' not in paths:
        raise KeyError("config['paths'] is missing 'init_graph_file'")

    # --- Simulation ---
    for key in ('max_time', 'time_step_minutes'):
        if key not in sim:
            raise KeyError(f"config['simulation'] is missing '{key}'")
    if sim['max_time'] <= 0:
        raise ValueError(f"config['simulation']['max_time'] must be > 0, got {sim['max_time']}")
    if sim['time_step_minutes'] <= 0:
        raise ValueError(
            f"config['simulation']['time_step_minutes'] must be > 0, got {sim['time_step_minutes']}")

    # --- Robot ---
    if 'policy' not in robot:
        raise KeyError("config['robot'] is missing 'policy'")
    valid_policies = ('slate_pro', 'streak', 'noop', 'hybrid', 'plain_llm',
                      'oracle', 'nopa')
    if robot['policy'] not in valid_policies:
        raise ValueError(
            f"config['robot']['policy'] must be one of {valid_policies}, got '{robot['policy']}'")


def apply_cli_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply ``--key.subkey value`` style overrides on top of *config*.

    Parameters
    ----------
    config : dict
        Base config loaded from YAML.
    overrides : list[str]
        Raw CLI tokens (remaining after argparse consumes its known args).
        Expected format: ``['--paths.init_graph_file', 'my_scene.json', ...]``
    """
    config = copy.deepcopy(config)
    i = 0
    while i < len(overrides):
        token = overrides[i]
        if token.startswith('--'):
            key = token[2:]  # strip leading '--'
            if i + 1 < len(overrides) and not overrides[i + 1].startswith('--'):
                val = _cast_value(overrides[i + 1])
                i += 2
            else:
                # Flag-style (no value) — treat as True
                val = True
                i += 1
            _set_nested(config, key, val)
        else:
            i += 1
    return config


# ======================================================================
# Simulation class
# ======================================================================

class Simulation:
    # Simulation minutes start at 6:00 AM — defined once in constants.py
    TIME_OFFSET_MINUTES = SIM_START_TIME_MINUTES

    def __init__(self, world, human_agent, robot_policy, config: dict,
                 logger: SimulationLogger | None = None):
        self.world = world
        self.human_agent = human_agent
        self.robot_policy = robot_policy
        self.config = config

        sim_cfg = config['simulation']
        self.current_time = 0
        self.max_time = sim_cfg['max_time']
        self.time_step_minutes = sim_cfg['time_step_minutes']
        self.verbose = sim_cfg['verbose']
        self.logger = logger

    @staticmethod
    def format_time(timestep: int) -> str:
        """Convert a simulation timestep to hh:mm AM/PM format (offset from 6:00 AM)."""
        total_minutes = Simulation.TIME_OFFSET_MINUTES + timestep
        total_minutes %= 24 * 60  # wrap around midnight
        hour = total_minutes // 60
        minute = total_minutes % 60
        period = 'AM' if hour < 12 else 'PM'
        display_hour = hour % 12
        if display_hour == 0:
            display_hour = 12
        return f'{display_hour}:{minute:02d} {period}'

    def run(self):
        current_day = 0  # For multi-day simulations; currently always 0
        script_path = getattr(self.human_agent, 'script_file', None)
        from agents.robot_policy_oracle import RobotPolicyOracle
        if isinstance(self.robot_policy, RobotPolicyOracle):
            self.robot_policy.reset(script_path=script_path)
        else:
            self.robot_policy.reset()
        while self.world.current_time < self.max_time:
            # Pass current world state so the human can check preconditions
            world_state = self.world.get_state()

            # Track where outcomes list starts so we can extract new entries
            prev_outcome_len = len(self.human_agent.action_outcomes)

            actions_human = self.human_agent.get_actions(
                self.current_time, world_state)
            action_robot = self.robot_policy.get_actions(
                self.current_time, self.world.node_map,
                world_state=world_state)

            if actions_human is not None:
                for action in actions_human:
                    self.world.set_human_activity(self.human_agent.get_human_activity())
                    if self.verbose:
                        print(f'Executing HUMAN action: {str(action)}')
                    self.world.execute_action(action)

            robot_move_success = None   # None = no action; True/False = succeeded/failed
            if action_robot is not None:
                if self.verbose:
                    print(f'Executing ROBOT action: {str(action_robot)}')
                if isinstance(action_robot, RobotMoveAction):
                    success, _ = self.world.execute_robot_move(action_robot)
                    robot_move_success = success
                    if not success:
                        print(f'[WARNING] Robot move failed: {action_robot}')
                else:
                    print(f'Warning: Robot action is not a RobotMoveAction: {action_robot}')
                    self.world.execute_action(action_robot)
                    robot_move_success = True

            # ---- Collect outcomes once (B7: was computed twice) ----
            new_outcomes = self.human_agent.action_outcomes[prev_outcome_len:]

            # ---- Print overview for monitoring ----
            time_str = self.format_time(self.current_time)
            robot_action_str = str(action_robot) if action_robot is not None else "None"
            if new_outcomes:
                human_action_strs = [o['action'] for o in new_outcomes]
                accommodation_strs = [o['outcome'] for o in new_outcomes]
                human_actions_display = ", ".join(human_action_strs) if human_action_strs else "None"
                accommodation_display = ", ".join(accommodation_strs) if accommodation_strs else "N/A"
            else:
                human_actions_display = "None" if actions_human is None else ", ".join([str(a) for a in actions_human])
                accommodation_display = "N/A"
            print(f"[{time_str:>8s}] Robot: {robot_action_str:30s} | Human: {human_actions_display:40s} | Accommodation: {accommodation_display}")

            # ---- Logging ----
            if self.logger is not None:
                human_action_strs = [o['action'] for o in new_outcomes] if new_outcomes else (
                    [str(a) for a in actions_human] if actions_human else None
                )
                human_outcome_strs = [o['outcome'] for o in new_outcomes] if new_outcomes else None
                robot_action_log = str(action_robot) if action_robot is not None else None
                self.logger.log_timestep(
                    t=self.current_time,
                    human_actions=human_action_strs,
                    human_outcomes=human_outcome_strs,
                    robot_action=robot_action_log,
                    robot_action_success=robot_move_success,
                )

            # ---- Verbose output (single block per timestep) ----
            if self.verbose:
                time_str = self.format_time(self.current_time)
                print(f'Time: {time_str} (step {self.current_time})')
                print(f'  Human Action: {", ".join([str(a) for a in actions_human]) if actions_human is not None else "None"}')
                print(f'  Robot Action: {str(action_robot)}')
                print_graph_difference(world_state, self.world.get_state())

            self.current_time += self.time_step_minutes
            self.world.current_time = self.current_time

        # ---- Finalize: end the day, save log, print summary ----
        if self.logger is not None:
            self.logger.end_day()
            log_path = self.logger.save()
            if log_path:
                print(f'\n[LOG] Saved simulation log to {log_path}')

        self._print_outcome_summary()
        return self.world.get_state()

    def _print_outcome_summary(self):
        """Print a summary of human action outcomes at the end of the episode."""
        outcomes = self.human_agent.action_outcomes
        if not outcomes:
            return
        counts = {'done_as_is': 0, 'corrected': 0, 'skipped': 0}
        for o in outcomes:
            counts[o['outcome']] = counts.get(o['outcome'], 0) + 1
        total = len(outcomes)
        print(f'\n--- Human Action Outcome Summary ---')
        print(f'  Total actions:  {total}')
        print(f'  Done as-is:     {counts["done_as_is"]}')
        print(f'  Corrected:      {counts["corrected"]}')
        print(f'  Skipped:        {counts["skipped"]}')


# ======================================================================
# Robot policy factory
# ======================================================================

def build_robot_policy(config: dict):
    """Instantiate the robot policy specified in *config*."""
    policy_name = config['robot']['policy']
    if policy_name == 'slate_pro':
        from agents.robot_policy_slatepro import RobotPolicySlatepro
        return RobotPolicySlatepro(config)
    elif policy_name == 'noop':
        from agents.robot_policy import RobotPolicyNoop
        return RobotPolicyNoop(config)
    elif policy_name == 'hybrid':
        from agents.robot_policy_hybrid import RobotPolicyHybrid
        return RobotPolicyHybrid(config)
    elif policy_name == 'oracle':
        from agents.robot_policy_oracle import RobotPolicyOracle
        return RobotPolicyOracle(config)
    elif policy_name == 'plain_llm':
        from agents.robot_policy_plainllm import RobotPolicyPlainLLM
        return RobotPolicyPlainLLM(config)
    elif policy_name == 'streak':
        from agents.robot_policy_streak import RobotPolicyStreak
        return RobotPolicyStreak(config)
    elif policy_name == 'nopa':
        from agents.robot_policy_nopa import RobotPolicyNOPA
        return RobotPolicyNOPA(config)
    else:
        raise ValueError(f'Invalid robot policy: {policy_name}')


def build_human_agent(config: dict, node_map, script_file: str,
                       accommodate_for_robot: bool = True):
    """Instantiate the human agent specified in ``config['human']['policy']``.

    Defaults to the scripted ``HumanAgent``. ``'llm'`` returns the
    :class:`environment.llm_human_agent.LLMHumanAgent`, which also forces
    the legacy vigilance roll off (the LLM handles that decision itself).
    """
    policy_name = config.get('human', {}).get('policy', 'scripted')
    if policy_name == 'scripted':
        return HumanAgent(config, node_map, script_file,
                          accommodate_for_robot=accommodate_for_robot)
    if policy_name == 'llm':
        # Force legacy vigilance off so the LLM is the sole arbiter.
        vig_cfg = config.setdefault('human', {}).setdefault('curiosity_vigilance', {})
        if vig_cfg.get('enabled'):
            print('[build_human_agent] policy=llm -> overriding '
                  'curiosity_vigilance.enabled to False')
        vig_cfg['enabled'] = False
        from environment.llm_human_agent import LLMHumanAgent
        return LLMHumanAgent(config, node_map, script_file,
                             accommodate_for_robot=accommodate_for_robot)
    raise ValueError(f'Invalid human policy: {policy_name}')


# ======================================================================
# Entry point
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the co-adaptation simulation.',
        epilog='Any additional --key.subkey value pairs override the YAML config '
               '(e.g. --robot.policy slate_pro --simulation.verbose true).',
    )
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help=f'Path to YAML config (default: {DEFAULT_CONFIG_PATH})')
    args, remaining = parser.parse_known_args()

    # Load base config, then apply CLI overrides
    config = load_config(args.config)
    config = apply_cli_overrides(config, remaining)
    validate_config(config)

    # Resolve paths
    paths = config['paths']
    sim_cfg = config['simulation']

    # Build components
    world = World(config)
    human_agent = HumanAgent(config, world.node_map)
    robot_policy = build_robot_policy(config)

    # Set up logger
    logger = None
    log_cfg = config.get('logging', {})
    if log_cfg.get('enabled', True):
        logger = SimulationLogger(config)

    simulation = Simulation(world, human_agent, robot_policy, config, logger=logger)
    simulation.run()
    print('Simulation completed successfully.')
