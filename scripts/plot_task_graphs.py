"""Visualize the hybrid policy's per-activity task graphs.

Renders each activity's branching action graph (agents/task_graph.py) to PNG
via the system `dot` binary. Nodes are actions (object -> destination), edge
labels are smoothed transition probabilities with raw counts; edge width
scales with probability; node fill shades from blue (action performed by the
HUMAN) to red (performed by the ROBOT), by the robot's share of the node mass
(count_robot / count) — purple in between.

Usage:
    python scripts/plot_task_graphs.py \
        --memory_dir results/cond23_.../hybrid_memory_A \
        [--activity breakfast_preparation_and_cleanup] [--min_p 0.05] \
        [--out_dir <memory_dir>/graph_viz]
"""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from agents.task_graph import TaskGraphSet, START, END


def _esc(s: str) -> str:
    return s.replace('"', '\\"')


def graph_to_dot(g, min_p: float, min_count: int = 1,
                 min_freq: float = 0.0) -> str:
    lines = [
        'digraph task_graph {',
        '  rankdir=LR;',
        '  graph [fontname="Helvetica", labelloc=t, '
        f'label="{_esc(g.activity_name)}  ({g.n_segments} segments)'
        '\\nnode fill: blue = human · red = robot (count_robot / count)", fontsize=20];',
        '  node [fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9];',
        '  START [shape=oval, style=filled, fillcolor="#7fc97f"];',
        '  END   [shape=oval, style=filled, fillcolor="#cccccc"];',
    ]
    hidden = {nid for nid, n in g.nodes.items()
              if n['count'] < min_count
              or n['count'] / max(g.n_segments, 1) < min_freq}
    for nid, n in g.nodes.items():
        if nid in hidden:
            continue
        cnt = n['count']
        rob = n.get('count_robot', 0) or 0
        frac = rob / cnt if cnt else 0.0          # 0 = all human, 1 = all robot
        # Diverging fill: light blue (human) <-> light red (robot), purple at
        # 50/50. Kept light (one channel held at 255-L) so black labels read.
        L = 140
        r = int(255 - L * (1.0 - frac))
        gch = int(255 - L)
        b = int(255 - L * frac)
        split = f'\\nrobot {rob:g} / human {cnt - rob:g}' if rob else ''
        lines.append(
            f'  n{nid} [shape=box, style="rounded,filled", '
            f'fillcolor="#{r:02x}{gch:02x}{b:02x}", '
            f'label="{_esc(n["obj_class"])}\\n→ {_esc(n["dest_class"])}\\n'
            f'x{cnt:g}{split}"];')

    def name(node_id):
        return {START: 'START', END: 'END'}.get(node_id, f'n{node_id}')

    for src in list(g.edge_counts):
        if src in hidden:
            continue
        probs = g.transition_probs(src)
        for dst, c in g.edge_counts[src].items():
            p = probs[dst]
            if p < min_p or dst in hidden:
                continue
            lines.append(
                f'  {name(src)} -> {name(dst)} '
                f'[label="{p:.2f} ({c})", penwidth={max(0.6, 5 * p):.2f}, '
                f'color="{"#444444" if p >= 0.5 else "#999999"}"];')
    lines.append('}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--memory_dir', required=True)
    ap.add_argument('--out_dir', default=None,
                    help='default: <memory_dir>/graph_viz')
    ap.add_argument('--activity', default=None,
                    help='render only this activity (default: all)')
    ap.add_argument('--min_p', type=float, default=0.05,
                    help='hide edges below this transition probability')
    ap.add_argument('--min_count', type=int, default=1,
                    help='hide action nodes observed fewer than N times '
                         '(declutters curation-noise stragglers)')
    ap.add_argument('--min_freq', type=float, default=0.0,
                    help='hide action nodes occurring in less than this '
                         'fraction of the activity\'s segments (e.g. 0.25)')
    ap.add_argument('--fmt', default='png', choices=['png', 'pdf', 'svg'])
    args = ap.parse_args()

    gs = TaskGraphSet.load(os.path.join(args.memory_dir, 'task_graphs.json'))
    out_dir = args.out_dir or os.path.join(args.memory_dir, 'graph_viz')
    os.makedirs(out_dir, exist_ok=True)

    names = [args.activity] if args.activity else sorted(gs.graphs)
    for name in names:
        g = gs.graphs[name]
        dot = graph_to_dot(g, args.min_p, args.min_count, args.min_freq)
        dot_path = os.path.join(out_dir, f'{name}.dot')
        img_path = os.path.join(out_dir, f'{name}.{args.fmt}')
        with open(dot_path, 'w') as f:
            f.write(dot)
        subprocess.run(['dot', f'-T{args.fmt}', dot_path, '-o', img_path],
                       check=True)
        print(f'{name:<40} {len(g.nodes):>3} nodes  -> {img_path}')


if __name__ == '__main__':
    main()
