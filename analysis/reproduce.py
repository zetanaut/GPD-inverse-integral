"""Verify inputs, recompute analysis tables, and regenerate all six figures."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from common import HERE, DEFAULT_OUT, verify_inputs, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--replay-checkpoints', action='store_true',
                        help='Also evaluate 54 saved models and replay the experimental NLO rows; requires the installed package and PyTorch.')
    args = parser.parse_args()
    out = args.out.resolve()
    if HERE == out or HERE in out.parents:
        parser.error('Choose an output directory outside the frozen analysis tree')
    count = verify_inputs()
    for script, dest in [('analyze_results.py', 'tables'), ('make_figures.py', 'figures')]:
        subprocess.run([sys.executable, str(HERE / script), '--out', str(out / dest)], check=True)
    if args.replay_checkpoints:
        subprocess.run([sys.executable, str(HERE / 'replay_checkpoints.py'), '--out', str(out / 'replay')], check=True)
    figures = sorted((out / 'figures').glob('fig*.pdf'))
    previews = sorted((out / 'figures').glob('fig*.png'))
    if len(figures) != 6 or len(previews) != 6 or any(p.stat().st_size < 1000 for p in figures + previews):
        raise RuntimeError('Expected six nonempty PDF figures and six PNG previews')
    result = dict(status='PASS', input_files_verified=count, figures=6,
                  checkpoint_replay=args.replay_checkpoints,
                  operations=['saved-result analysis', 'figure generation'] +
                  (['saved-checkpoint evaluation'] if args.replay_checkpoints else []),
                  no_training=True, no_new_noise=True)
    write_json(out / 'REPRODUCTION_CHECK.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
