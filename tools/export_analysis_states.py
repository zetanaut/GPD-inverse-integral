"""Export the bound original checkpoints as portable NPZ state arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True,
                        help='Original research workspace containing the paths recorded in the checkpoint index')
    parser.add_argument('--out', type=Path, required=True,
                        help='Destination for exported states; does not change the committed index or input manifest')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    index = json.loads((root / 'analysis/checkpoints/index.json').read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for record in index['models']:
        src = args.workspace / record['source_checkpoint']
        if hashlib.sha256(src.read_bytes()).hexdigest() != record['source_sha256']:
            raise ValueError('Original checkpoint hash mismatch: ' + str(src))
        checkpoint = torch.load(src, map_location='cpu', weights_only=True)
        state = {k: v.detach().cpu().numpy() for k, v in checkpoint['state_dict'].items()}
        if sum(v.size for v in state.values()) != record['parameter_count']:
            raise ValueError('Unexpected state size')
        if 'flat' in checkpoint:
            np.testing.assert_array_equal(np.concatenate([v.ravel() for v in state.values()]), checkpoint['flat'].numpy())
        with np.load(root / 'analysis' / record['path'], allow_pickle=False) as bundled:
            if set(bundled.files) != set(state):
                raise ValueError('State keys differ')
            for key in state:
                np.testing.assert_array_equal(state[key], bundled[key])
        dest = args.out / Path(record['path']).name
        np.savez_compressed(dest, **state)
    print('Exported and compared', len(index['models']), 'model states.')


if __name__ == '__main__':
    main()
