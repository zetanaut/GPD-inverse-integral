#!/usr/bin/env python3
"""Fit the predeclared ReH surface and compare it with saved central13 values."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOURCE = PROJECT / "GPD_ReH_experimental_run/inputs/final_summary_by_DNN.csv"
BACKUP = PROJECT / "GPD_experimental_closeout/evidence/final_summary_by_DNN.csv"
PREPARED = PROJECT / "GPD_ReH_experimental_run/prepared/observations.csv"
INPUT_SUMMARY = PROJECT / "GPD_ReH_experimental_run/prepared/INPUT_SUMMARY.json"
CONTRACT = PROJECT / "GPD_ReH_experimental_run/CENTRAL_CONTRACT.json"
INDEX = PROJECT / "GPD_experimental_closeout/models/index.json"
PREDICTIONS = PROJECT / "GPD_experimental_closeout/results/ReH_predictions.csv"
CENTRAL_PT = PROJECT / "GPD_experimental_closeout/models/central13.pt"
CENTRAL_NPZ = PROJECT / "GPD_experimental_closeout/models/central13.npz"
EXPECTED_SOURCE = "8d7274db9dd7e4fcc3ab84c0230701242274e42868f9c3e66e3d61a1a6838696"
PROTO = json.loads((HERE / "FIT_PROTOCOL.json").read_text())
SEED = PROTO["seed"]
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def same(a, b, name):
    if not np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=0, atol=1e-11):
        raise ValueError(f"row mismatch: {name}")


def audit():
    refs = {str(p.relative_to(PROJECT)): sha(p) for p in
            [SOURCE, BACKUP, PREPARED, INPUT_SUMMARY, CONTRACT, INDEX, PREDICTIONS, CENTRAL_PT, CENTRAL_NPZ]}
    assert sha(SOURCE) == EXPECTED_SOURCE == sha(BACKUP)
    source = pd.read_csv(SOURCE)
    obs = pd.read_csv(PREPARED)
    pred = pd.read_csv(PREDICTIONS)
    assert len(source) == 189 and len(obs) == 160
    assert source.DNN.value_counts().to_dict() == {"CDNN": 101, "FQDNN": 88}
    cut = (source.QQ > 1.5) & ((-source.t).abs() / source.QQ <= 0.25)
    assert cut.sum() == 160
    assert obs.row_id.tolist() == [f"reh:{i:04d}" for i in (obs.source_line.to_numpy() - 2)]
    assert (obs.paper_cut_pass == 1).all()
    assert obs.source_line.tolist() == (np.flatnonzero(cut) + 2).tolist()
    s = source.iloc[obs.source_line.to_numpy() - 2].reset_index(drop=True)
    assert obs.method.tolist() == s.DNN.tolist()
    same(obs.source_set, s["Set #"], "set")
    for a, b in [("k_GeV", "k"), ("Q2_GeV2", "QQ"), ("xB", "x_b"),
                 ("t_GeV2", "t"), ("ReH_source", "Pred ReH"),
                 ("sigma_reported", "Pred ReH Std")]:
        same(obs[a], s[b], a)
    same(obs.xi, obs.xB / (2 - obs.xB), "xi")
    same(obs.ReH_internal, -obs.ReH_source, "source/internal sign")
    assert (obs.sigma_reported > 0).all()
    central = pred[pred.state == "central13"].reset_index(drop=True)
    assert len(central) == 160 and central.row_id.tolist() == obs.row_id.tolist()
    assert central.method.tolist() == obs.method.tolist()
    for a, b in [("source_set", "source_set"), ("xi", "xi"),
                 ("Q2_GeV2", "Q2_GeV2"), ("t_GeV2", "t_GeV2"),
                 ("ReH_source", "ReH_source"), ("sigma_reported", "sigma_reported_combined")]:
        same(obs[a], central[b], f"central13 {a}")
    index = json.loads(INDEX.read_text())["central13"]
    assert sha(CENTRAL_PT) == index["checkpoint_sha256"]
    assert sha(CENTRAL_NPZ) == index["npz_sha256"]
    assert json.loads(INPUT_SUMMARY.read_text())["source_sha256"] == EXPECTED_SOURCE
    assert json.loads(CONTRACT.read_text())["input"]["sha256"] == EXPECTED_SOURCE
    audit_result = {"status": "verified", "source_rows": 189, "selected_rows": 160,
                    "excluded_rows": 29, "source_method_counts": {"CDNN": 101, "FQDNN": 88},
                    "selected_method_counts": obs.method.value_counts().to_dict(),
                    "source_sign": "ReH_source = -ReH_internal; comparison and fit use source sign",
                    "central_checkpoint": "GPD_experimental_closeout/models/central13.pt",
                    "central_prediction_column": "ReH_fitted_source", "sha256": refs}
    (HERE / "AUDIT.json").write_text(json.dumps(audit_result, indent=2) + "\n")
    return obs, central


def features(obs):
    return np.column_stack([obs.xi.to_numpy(), obs.t_GeV2.to_numpy(), np.log(obs.Q2_GeV2.to_numpy())])


def scales(x, y, ids):
    xm, xs = x[ids].mean(axis=0), x[ids].std(axis=0)
    ym, ys = y[ids].mean(), y[ids].std()
    assert np.all(xs > 0) and ys > 0
    return xm, xs, ym, ys


class Surface(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(3, 32), torch.nn.Tanh(),
                                       torch.nn.Linear(32, 32), torch.nn.Tanh(),
                                       torch.nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def setup(x, y, sigma, ids):
    xm, xs, ym, ys = scales(x, y, ids)
    xt = torch.tensor((x - xm) / xs)
    yt = torch.tensor((y - ym) / ys)
    st = torch.tensor(sigma / ys)
    torch.manual_seed(SEED)
    return Surface(), xt, yt, st, (xm, xs, ym, ys)


def fit(model, xt, yt, st, train_ids, n_epochs, val_ids=None):
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    tr = torch.tensor(train_ids, dtype=torch.long)
    va = torch.tensor(val_ids, dtype=torch.long) if val_ids is not None else None
    best, best_epoch, patience, best_state = math.inf, 0, 0, None
    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(xt[tr])
        data_loss = torch.mean(((pred - yt[tr]) / st[tr]) ** 2)
        penalty = 0.0001 * sum(torch.mean(p.square()) for name, p in model.named_parameters() if name.endswith("weight"))
        loss = data_loss + penalty
        loss.backward()
        opt.step()
        if va is not None:
            model.eval()
            with torch.no_grad():
                vmse = torch.mean(((model(xt[va]) - yt[va]) / st[va]) ** 2).item()
            history.append((epoch, float(data_loss.detach()), vmse))
            if vmse < best - 1e-5:
                best, best_epoch, patience = vmse, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 200:
                    break
    if va is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def stats(resid, sigma):
    pull = resid / sigma
    return {"mean_abs_ReH": float(np.mean(np.abs(resid))),
            "median_abs_ReH": float(np.median(np.abs(resid))),
            "rms_ReH": float(np.sqrt(np.mean(resid ** 2))),
            "mean_abs_pull": float(np.mean(np.abs(pull))),
            "rms_pull": float(np.sqrt(np.mean(pull ** 2))),
            "count_abs_pull_gt_3": int((np.abs(pull) > 3).sum())}


def main():
    obs, central = audit()
    x = features(obs)
    y, sigma = obs.ReH_source.to_numpy(), obs.sigma_reported.to_numpy()
    p = central.ReH_fitted_source.to_numpy()
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(p).all()
    perm = np.random.default_rng(SEED).permutation(len(obs))
    val, train = perm[:32], perm[32:]
    model, xt, yt, st, split_scale = setup(x, y, sigma, train)
    model, best_epoch, history = fit(model, xt, yt, st, train, 2000, val)
    with torch.no_grad():
        split_predictions = model(xt).numpy() * split_scale[3] + split_scale[2]
    val_stats = stats(split_predictions[val] - y[val], sigma[val])
    train_stats = stats(split_predictions[train] - y[train], sigma[train])
    assert 1 <= best_epoch <= 2000
    model, xt, yt, st, full_scale = setup(x, y, sigma, np.arange(160))
    model, _, _ = fit(model, xt, yt, st, np.arange(160), best_epoch)
    model.eval()
    with torch.no_grad():
        smooth = model(xt).numpy() * full_scale[3] + full_scale[2]
    assert np.isfinite(smooth).all()
    checkpoint = {"state_dict": model.state_dict(), "input_mean": full_scale[0].tolist(),
                  "input_std": full_scale[1].tolist(), "target_mean": float(full_scale[2]),
                  "target_std": float(full_scale[3]), "feature_order": ["xi", "t_GeV2", "log(Q2_GeV2)"],
                  "best_validation_epoch": best_epoch, "protocol_sha256": sha(HERE / "FIT_PROTOCOL.json"),
                  "source_sha256": EXPECTED_SOURCE, "prepared_sha256": sha(PREPARED), "seed": SEED}
    torch.save(checkpoint, HERE / "ReH_fitted_central_surface.pt")
    result = obs[["row_id", "source_line", "source_set", "method", "k_GeV", "xi", "t_GeV2", "Q2_GeV2", "ReH_source", "sigma_reported"]].copy()
    result["ReH_smooth_source"] = smooth
    result["ReH_central13_source"] = p
    for label, r in [("smooth_minus_local", smooth-y), ("central13_minus_local", p-y),
                     ("central13_minus_smooth", p-smooth)]:
        result[f"{label}_signed"] = r
        result[f"{label}_abs"] = np.abs(r)
        result[f"{label}_pull"] = r / sigma
        result[f"{label}_abs_pull"] = np.abs(r / sigma)
    result.to_csv(HERE / "comparison_160.csv", index=False, float_format="%.17g")
    metrics = {"selected_validation_epoch": best_epoch,
               "split": {"seed": SEED, "training_rows": train.tolist(), "validation_rows": val.tolist(),
                         "training_at_selected_epoch": train_stats, "validation_at_selected_epoch": val_stats,
                         "epochs_executed": len(history)},
               "full_surface_vs_selected_local": stats(smooth-y, sigma),
               "saved_central13_vs_selected_local": stats(p-y, sigma),
               "saved_central13_vs_full_surface": stats(p-smooth, sigma),
               "same_sigma_in_all_pulls": True,
               "interpretation": "All-row final-surface residuals are in-sample; validation is from the separate 128/32 split stage."}
    (HERE / "METRICS.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with open(HERE / "fit_history.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "training_weighted_mse", "validation_weighted_mse"]); w.writerows(history)
    plot(result, train, val)
    print(json.dumps({"best_epoch": best_epoch, "validation": val_stats,
                      "smooth_vs_local": metrics["full_surface_vs_selected_local"],
                      "central13_vs_local": metrics["saved_central13_vs_selected_local"],
                      "central13_vs_smooth": metrics["saved_central13_vs_full_surface"]}, indent=2))


def plot(d, train, val):
    y, sigma = d.ReH_source.to_numpy(), d.sigma_reported.to_numpy()
    smooth, p = d.ReH_smooth_source.to_numpy(), d.ReH_central13_source.to_numpy()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True, layout="constrained")
    ids = np.arange(len(d))
    axs[0].errorbar(ids, y, yerr=sigma, fmt=".", color="0.4", alpha=.55, label="selected local ± quoted σ")
    axs[0].plot(ids, smooth, ".", ms=4, color="#047d79", label="fitted central surface")
    axs[0].plot(ids, p, ".", ms=4, color="#b64b2e", label="saved central13 GPD")
    axs[0].set_ylabel("ReH, source convention")
    axs[0].legend(ncol=3, fontsize=9)
    axs[1].plot(ids, (smooth-y)/sigma, ".", ms=4, color="#047d79", label="surface − local")
    axs[1].plot(ids, (p-y)/sigma, ".", ms=4, color="#b64b2e", label="central13 − local")
    axs[1].axhline(0, color="0.3", lw=.7)
    axs[1].axhline(3, color="0.6", lw=.7, ls="--")
    axs[1].axhline(-3, color="0.6", lw=.7, ls="--")
    axs[1].set(xlabel="prepared row order (retained source identities)", ylabel="residual / original quoted σ")
    axs[1].legend()
    fig.savefig(HERE / "comparison_by_row.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, z, label, color in [(axs[0], smooth, "fitted central surface", "#047d79"),
                                 (axs[1], p, "saved central13 GPD", "#b64b2e")]:
        ax.scatter(y, z, c=color, s=15, alpha=.75)
        lo, hi = min(y.min(), z.min()), max(y.max(), z.max())
        ax.plot([lo, hi], [lo, hi], color="0.3", lw=.8)
        ax.set(xlabel="selected local ReH", ylabel=label + " ReH")
    fig.savefig(HERE / "predicted_vs_local.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
