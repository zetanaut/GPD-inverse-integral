"""Scalar/array bridge to the captured reduced-BKM source, not a new formula.

Only the calculator classes are compiled; Keras training and plotting are not
imported. A private NumPy namespace implements their scalar TensorFlow calls.
Float64 mode promotes cast arithmetic explicitly, retaining a source-precision
control. It does not install or shadow a TensorFlow module.
"""
from __future__ import annotations
import ast, math, types
from pathlib import Path
import numpy as np


def load_calculator(path: str | Path, float64: bool = True):
    tree = ast.parse(Path(path).read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name in {"BHDVCStf", "F_calc", "FFs"}]
    if {n.name for n in classes} != {"BHDVCStf", "F_calc", "FFs"}:
        raise ValueError("Expected calculator classes are absent")
    for node in classes:
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                child.decorator_list = []
    def split(value, num_or_size_splits, axis=0):
        if isinstance(num_or_size_splits, int):
            return np.split(value, num_or_size_splits, axis=axis)
        return np.split(value, np.cumsum(num_or_size_splits)[:-1], axis=axis)
    tf = types.SimpleNamespace(
        constant=lambda x: np.asarray(x, dtype=np.float64), sqrt=np.sqrt,
        cos=np.cos, pow=np.power, get_static_value=np.asarray,
        cast=lambda x, dtype: np.asarray(x, dtype=np.float64 if float64 else dtype),
        split=split)
    namespace = {"tf": tf, "np": np, "math": math}
    module = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["F_calc"]()


def evaluate(calc, kinematics: np.ndarray, cffs: np.ndarray) -> np.ndarray:
    """Canonical columns [k_GeV, Q2_GeV2, xB, t_GeV2, phi_deg]."""
    kinematics = np.asarray(kinematics, dtype=float)
    cffs = np.asarray(cffs, dtype=float)
    if kinematics.ndim != 2 or kinematics.shape[1] != 5:
        raise ValueError("kinematics must have shape (N,5)")
    if cffs.shape == (4,):
        cffs = np.broadcast_to(cffs, (len(kinematics), 4))
    if cffs.shape != (len(kinematics), 4):
        raise ValueError("CFFs must have shape (4,) or (N,4)")
    k, q2, xb, t, phi = kinematics.T
    result = np.asarray(calc.fn_1([phi, q2, xb, t, k], cffs.T), dtype=float)
    if result.shape != (len(kinematics),) or not np.all(np.isfinite(result)):
        raise ValueError("Nonfinite/incorrect-shape reduced BKM evaluation")
    return result


def affine_design(calc, kinematics: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    n = len(kinematics)
    b = evaluate(calc, kinematics, np.zeros(4))
    A = np.column_stack([evaluate(calc, kinematics, np.eye(4)[j]) - b for j in range(4)])
    return b, A
