#!/usr/bin/env python3.8
"""
evaluate_all.py — Step 7 Head-to-Head Evaluation: Policy A vs B vs C.

Runs 3 policies × 5 held-out conditions × 20 episodes = 300 evaluation episodes.
Results are saved to results/comparison_data.json after each cell (crash-safe).
Chart and statistical summary are generated after all cells complete.

Prerequisites (inside Docker):
    pip install matplotlib scipy

Usage:
    python3.8 scripts/evaluate_all.py           # full 300-episode run (~65 min)
    python3.8 scripts/evaluate_all.py --episodes 5  # quick smoke test (~16 min)

Outputs:
    results/comparison_data.json  — per-episode data for all 15 cells
    results/comparison_chart.png  — grouped bar chart (displacement + fall rate)

Resume: if interrupted, re-run the same command — completed cells are skipped.
"""

import sys
import os
import json
import argparse
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from rl_env.rospug_env import DT
from rl_env.rospug_env_dr import RosPugEnvDR
from stable_baselines3 import PPO

_ROOT        = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RESULTS_DIR  = os.path.join(_ROOT, 'results')
DATA_PATH    = os.path.join(RESULTS_DIR, 'comparison_data.json')
CHART_PATH   = os.path.join(RESULTS_DIR, 'comparison_chart.png')

# Policy A's nominal fixed-physics performance (reference line on chart)
POLICY_A_NOMINAL_DISP = 3.244

POLICIES = {
    'A': os.path.join(_ROOT, 'checkpoints', 'policy_A_500k.zip'),
    'B': os.path.join(_ROOT, 'checkpoints', 'policy_B_500k.zip'),
    'C': os.path.join(_ROOT, 'checkpoints', 'policy_C_500k.zip'),
}

# erp=0.55 = midpoint of training range [0.3, 0.8]; nominal for non-erp conditions
CONDITIONS = [
    {
        'name':    'Light body',
        'short':   'light_body',
        'mass':    0.80,
        'latency': 0.000,
        'cfm':     0.000,
        'erp':     0.55,
    },
    {
        'name':    'Heavy body',
        'short':   'heavy_body',
        'mass':    1.20,
        'latency': 0.000,
        'cfm':     0.000,
        'erp':     0.55,
    },
    {
        'name':    'High latency',
        'short':   'high_latency',
        'mass':    1.00,
        'latency': 0.015,
        'cfm':     0.000,
        'erp':     0.55,
    },
    {
        'name':    'Slippery surface',
        'short':   'slippery',
        'mass':    1.00,
        'latency': 0.000,
        'cfm':     0.001,
        'erp':     0.55,
    },
    {
        'name':    'Worst-case combo',
        'short':   'worst_case',
        'mass':    1.20,
        'latency': 0.015,
        'cfm':     0.000,
        'erp':     0.55,
    },
]


def _cell_key(policy_label: str, cond_short: str) -> str:
    return f'{policy_label}__{cond_short}'


def run_cell(env: RosPugEnvDR, policy_label: str, cond: dict, n_episodes: int) -> list:
    """Run one policy × one condition cell; returns list of per-episode dicts."""
    # Update ranges to the fixed condition values (lo==hi → always samples exactly x)
    v = cond['mass']
    env._mass_range    = (v,              v)
    env._latency_range = (cond['latency'], cond['latency'])
    env._cfm_range     = (cond['cfm'],    cond['cfm'])
    env._erp_range     = (cond['erp'],    cond['erp'])

    model = PPO.load(POLICIES[policy_label])

    _FALL_THRESH = 0.7  # must match rospug_env.py FALL_THRESH

    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        # obs[24]=roll, obs[25]=pitch; retry if robot starts already fallen
        for _retry in range(3):
            if abs(float(obs[24])) < _FALL_THRESH and abs(float(obs[25])) < _FALL_THRESH:
                break
            print(f'      [RETRY] Start state fallen (roll={obs[24]:.2f} pitch={obs[25]:.2f}), re-resetting...',
                  flush=True)
            obs, _ = env.reset()
        ep_reward = 0.0
        x_disp    = 0.0
        fell      = False
        steps     = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            x_disp    += info['vx'] * DT
            steps     += 1
            if terminated:
                fell = True
            if terminated or truncated:
                break

        episodes.append({
            'displacement': round(x_disp, 4),
            'fell':         fell,
            'reward':       round(ep_reward, 4),
            'steps':        steps,
        })
        status = 'FELL    ' if fell else 'SURVIVED'
        print(f'    [ep {ep+1:2d}/{n_episodes}] {status}  x={x_disp:+6.2f} m  '
              f'steps={steps}', flush=True)

    return episodes


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def print_summary_table(data: dict) -> None:
    try:
        from scipy.stats import ttest_ind
        scipy_ok = True
    except ImportError:
        scipy_ok = False
        print('[WARN] scipy not found — p-values omitted. Install with: pip install scipy')

    header = f"{'Condition':<20} {'Policy':<6} {'Mean disp (m)':<16} {'Std':<8} {'Fall%':<8}"
    if scipy_ok:
        header += f" {'p(B vs C)':<12}"
    print('\n' + '=' * 75)
    print('Step 7 — Head-to-Head Summary')
    print('=' * 75)
    print(header)
    print('-' * 75)

    for cond in CONDITIONS:
        for i, label in enumerate(('A', 'B', 'C')):
            key  = _cell_key(label, cond['short'])
            eps  = data.get(key, [])
            disps = [e['displacement'] for e in eps]
            falls = [e['fell'] for e in eps]
            m    = _mean(disps)
            s    = _std(disps)
            fr   = _mean(falls) * 100.0
            row  = f"{cond['name'] if i==0 else '':<20} {label:<6} {m:>+8.3f} m      {s:>6.3f}   {fr:>5.1f}%"
            if scipy_ok and i == 2:
                b_disps = [e['displacement'] for e in data.get(_cell_key('B', cond['short']), [])]
                c_disps = disps
                if len(b_disps) >= 2 and len(c_disps) >= 2:
                    _, p = ttest_ind(b_disps, c_disps, equal_var=False)
                    row += f'   p={p:.3f}{"*" if p < 0.05 else ""}'
                else:
                    row += '   p=n/a'
            print(row)
        print()

    print(f'Reference: Policy A nominal (fixed physics) = {POLICY_A_NOMINAL_DISP:.3f} m')
    print('* p < 0.05 (Welch\'s t-test, B vs C)')
    print('=' * 75)


def generate_chart(data: dict) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print('[WARN] matplotlib not found — chart skipped. Install with: pip install matplotlib')
        return

    cond_names  = [c['name'] for c in CONDITIONS]
    cond_shorts = [c['short'] for c in CONDITIONS]
    labels      = ('A', 'B', 'C')
    colours     = ('#4878CF', '#F28E2B', '#59A14F')   # blue, orange, green

    n_conds = len(CONDITIONS)
    x       = np.arange(n_conds)
    width   = 0.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.40)

    # ── Subplot 1: mean displacement ────────────────────────────────────────────
    for i, (label, colour) in enumerate(zip(labels, colours)):
        means = []
        errs  = []
        for short in cond_shorts:
            eps   = data.get(_cell_key(label, short), [])
            disps = [e['displacement'] for e in eps]
            means.append(_mean(disps))
            errs.append(_std(disps))
        offset = (i - 1) * width
        ax1.bar(x + offset, means, width, label=f'Policy {label}',
                color=colour, alpha=0.85, yerr=errs, capsize=4,
                error_kw={'elinewidth': 1.2, 'ecolor': 'grey'})

    ax1.axhline(y=POLICY_A_NOMINAL_DISP, color='black', linestyle='--', linewidth=1.0,
                label=f'Policy A nominal ({POLICY_A_NOMINAL_DISP:.3f} m)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cond_names, fontsize=9)
    ax1.set_ylabel('Mean x-displacement (m)')
    ax1.set_title('Head-to-Head Evaluation — Mean Displacement (±1σ)\nPolicy A vs B vs C on 5 Held-Out Conditions')
    ax1.legend(fontsize=9)
    ax1.set_ylim(bottom=0)

    # ── Subplot 2: fall rate ─────────────────────────────────────────────────────
    for i, (label, colour) in enumerate(zip(labels, colours)):
        fall_rates = []
        for short in cond_shorts:
            eps  = data.get(_cell_key(label, short), [])
            fr   = _mean([e['fell'] for e in eps]) * 100.0 if eps else 0.0
            fall_rates.append(fr)
        offset = (i - 1) * width
        ax2.bar(x + offset, fall_rates, width, label=f'Policy {label}',
                color=colour, alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_names, fontsize=9)
    ax2.set_ylabel('Fall rate (%)')
    ax2.set_ylim(0, 110)
    ax2.set_title('Fall Rate by Condition')
    ax2.legend(fontsize=9)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(CHART_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[INFO] Chart saved to {CHART_PATH}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Step 7: A vs B vs C head-to-head evaluation on held-out conditions.'
    )
    parser.add_argument('--episodes', type=int, default=20,
                        help='Episodes per policy × condition cell (default: 20)')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load existing results (resume support)
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH) as f:
            data = json.load(f)
        print(f'[INFO] Resuming from {DATA_PATH} ({len(data)} cells already done)')
    else:
        data = {}

    total_cells    = len(POLICIES) * len(CONDITIONS)
    completed_cells = 0

    print('=' * 70)
    print('Step 7 — Head-to-Head Evaluation: Policy A vs B vs C')
    print(f'  Policies   : {list(POLICIES.keys())}')
    print(f'  Conditions : {len(CONDITIONS)}')
    print(f'  Episodes   : {args.episodes} per cell')
    print(f'  Total eps  : {len(POLICIES) * len(CONDITIONS) * args.episodes}')
    print('=' * 70)

    # One env instance for the entire run — avoids rospy re-init between cells
    env = RosPugEnvDR(node_name='eval_all')
    try:
        for label in ('A', 'B', 'C'):
            for cond in CONDITIONS:
                key = _cell_key(label, cond['short'])
                if key in data:
                    completed_cells += 1
                    print(f'[SKIP] Policy {label} / {cond["name"]} — already done '
                          f'({len(data[key])} episodes)')
                    continue

                completed_cells += 1
                print(f'\n[{completed_cells}/{total_cells}] Policy {label} / {cond["name"]}')
                print(f'  mass={cond["mass"]:.2f}  latency={cond["latency"]*1000:.1f}ms  '
                      f'cfm={cond["cfm"]:.4f}  erp={cond["erp"]:.2f}')

                episodes = run_cell(env, label, cond, args.episodes)
                data[key] = episodes

                # Save after every cell so a crash loses at most one cell
                with open(DATA_PATH, 'w') as f:
                    json.dump(data, f, indent=2)

                disps = [e['displacement'] for e in episodes]
                falls = [e['fell'] for e in episodes]
                print(f'  → mean disp={_mean(disps):+.3f} m  std={_std(disps):.3f}  '
                      f'fall rate={_mean(falls)*100:.0f}%')
    finally:
        env.close()

    print('\n[INFO] All cells complete.')
    print_summary_table(data)
    generate_chart(data)


if __name__ == '__main__':
    main()
