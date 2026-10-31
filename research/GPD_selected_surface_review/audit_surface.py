#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from surface_tools import audit

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Read-only selected ReH surface replay; no fitting')
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--surface', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    surface = a.surface or a.project / 'completion_runs/ReH_selected_surface'
    result = audit(surface, a.project)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ['status','rows','artifact_hashes_verified','source_reference_hashes_verified','maximum_numpy_prediction_difference']}, indent=2))
