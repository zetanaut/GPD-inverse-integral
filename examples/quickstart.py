"""Evaluate current interfaces without fitting or constructing evolution matrices."""
from pathlib import Path
import json
import torch
from gpd_reh.model import ExperimentalFiLM
from gpd_reh.moments import FixedTMoments
from gpd_reh.bank import ExperimentalForward, preflight

def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    data = root / 'data/experimental'
    meta = json.loads((data / 'prepared/INPUT_SUMMARY.json').read_text())
    torch.manual_seed(31)
    model = ExperimentalFiLM(meta['t_mean'], meta['t_std_population']).double()
    forward = ExperimentalForward(data / 'prepared', data / 'banks')
    with torch.no_grad():
        predictions = forward(model)
        moments = FixedTMoments([-.363], order=16)
        residual = float(moments(model))
    print(json.dumps({'role': 'fresh random model interface demo; no fit',
                     'parameters': sum(p.numel() for p in model.parameters()),
                     'bank_status': preflight(data/'prepared', data/'banks'),
                     'prediction_shape': list(predictions.shape),
                     'predictions_finite': bool(torch.isfinite(predictions).all()),
                     'moment_residual_order16': residual}, indent=2))

if __name__ == '__main__':
    main()
