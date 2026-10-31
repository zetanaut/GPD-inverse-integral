"""Tiny synthetic tests. No GPD data/model, fit, or kernel is constructed."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from p2b_policy import CAPSULE, CASE, check_restart
CASES = {CASE: .03}

NUMERICAL = '--fixtures' in sys.argv
if NUMERICAL:
    sys.argv.remove('--fixtures')
    sys.path.insert(0, str(CAPSULE / 'runtime'))
    import core
    core.setup()
    import numpy as np
    import torch
    import resume_support
    import run_fit
    solver_spec = importlib.util.spec_from_file_location('toy_inherited_solver', CAPSULE / 'project/completion_runs/NLO_constrained_solver/solver.py')
    original_solver = importlib.util.module_from_spec(solver_spec)
    solver_spec.loader.exec_module(original_solver)
    # Extract only the original Adam function; importing solver_binding would load scientific models.
    source_path = CAPSULE / 'project/GPD_three_stage_closure/closure/solver_binding.py'
    source_text = source_path.read_text()
    node = next(n for n in ast.parse(source_text).body if isinstance(n, ast.FunctionDef) and n.name == 'adam400')
    # Real filename and preserved line offsets allow the unchanged resume adapter's inspect.getsource.
    code = ast.Module(body=[node], type_ignores=[])
    namespace = {'torch': torch}
    exec(compile(code, str(source_path), 'exec'), namespace)
    adam400 = namespace['adam400']

@unittest.skipUnless(NUMERICAL, 'explicit synthetic-fixture test')
class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, {'SLURM_JOB_ID': 'synthetic_fixture'})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.tmp = tempfile.TemporaryDirectory(prefix='GPD_P2B_test_')
        self.root = Path(self.tmp.name)
        run_fit.STOP = False

    def tearDown(self):
        self.tmp.cleanup()

    def test_adam_resume_exact_original_400_updates(self):
        problem = types.SimpleNamespace(x0=torch.tensor([1., -.5]), objective=lambda x: ((x - .25)**2).sum())
        expected, _ = resume_support.run_adam(problem, adam400, self.root / 'uninterrupted.pt')
        original_save = resume_support.atomic_torch
        path = self.root / 'paused.pt'
        class Interrupted(Exception):
            pass
        def save_then_stop(path, obj):
            original_save(path, obj)
            if obj['completed_updates'] == 137:
                raise Interrupted()
        with patch.object(resume_support, 'atomic_torch', save_then_stop):
            with self.assertRaises(Interrupted):
                resume_support.run_adam(problem, adam400, path)
        saved = torch.load(path, weights_only=False)
        self.assertEqual(saved['remaining_updates'], 263)
        actual, _ = resume_support.run_adam(problem, adam400, path)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(torch.load(path, weights_only=False)['completed_updates'], 400)

    def test_driver_resume_exact_outer_state_and_objective(self):
        self.exercise_driver(interrupt='solver')

    def test_driver_resume_exact_adam_state_and_objective(self):
        self.exercise_driver(interrupt='adam')

    def test_terminal_pause_finalizes_without_more_updates(self):
        self.exercise_driver(interrupt='terminal')

    def exercise_driver(self, interrupt):
        class ToyProblem:
            def __init__(self):
                self.x0 = torch.tensor([.8, -.4, .6])
            def parameters(self, x):
                return {'synthetic': x}
        class ToyOverlay:
            def __init__(self, base, information=None, **unused):
                self.base, self.strength, self.x0 = base, .03, base.x0
            def parameters(self, x):
                return self.base.parameters(x)
            def objective(self, x):
                return ((x - .2)**2).sum() + self.strength * (x*x).sum()
            def named_components(self, x):
                data, reg = ((x - .2)**2).sum(), (x*x).sum()
                zero = x.new_tensor(0.)
                return dict(J_CFF=data, J_poly=zero, J_PDF=self.strength*reg,
                            chi2_PDF=160*self.strength*reg, J_P1=data, J_joint=self.objective(x))
        class Oracle:
            def __init__(self, p, d, x):
                self.p, self.x0, self.names = p, x.numpy().copy(), ['synthetic_constant']
                self.counts = dict(evaluations=0, objective_gradients=0, residual_jacobians=0, constraint_jacobian_builds=0, constraint_jacobian_rows=0)
                self.timers = {}
            def values(self, x):
                self.counts['evaluations'] += 1
                residual = np.r_[x - .2, self.p.strength**.5*x]
                return dict(J=float(residual@residual), c=np.array([-1.]), components=np.array([float(((x-.2)**2).sum()), 0., float(self.p.strength*(x*x).sum())]), residual=residual)
            def derivatives(self, x, working):
                self.counts['objective_gradients'] += 1
                self.counts['residual_jacobians'] += 1
                self.counts['constraint_jacobian_builds'] += 1
                return 2*(x-.2)+2*self.p.strength*x, np.r_[np.eye(3), self.p.strength**.5*np.eye(3)], np.zeros((len(working),3))
        def kkt(g, *unused):
            return {'pass': bool(np.linalg.norm(g) < 1e-8)}
        def isolated(oracle):
            return original_solver, oracle, {}, dict(original_solver.SETTINGS, max_iterations=60,
                max_evaluations=12000, initial_radius=.01, optimization_seconds=float('inf'))
        def restore(p, d, adam):
            # Synthetic fixture preserves the call/selection interface; no real restoration is claimed.
            return (float(p.objective(p.x0)), 'random', p.x0.clone()), [{'fixture': 'synthetic'}]
        original_loader = core.load_module
        def loader(name, path):
            if path.name != 'finalize.py':
                return original_loader(name, path)
            module = types.SimpleNamespace()
            def finalize(case, p, d, out, complete):
                self.assertIsInstance(p, ToyOverlay)
                candidate = complete['candidates_by_supplied_J'][0]
                ck = torch.load(out / candidate['path'], weights_only=True)
                self.assertAlmostEqual(ck['J'], float(p.objective(ck['flat'])), places=13)
                core.save(out / 'final.pt', ck)
                core.write(out / 'SELECTION.json', {'scored': candidate})
                module.write(out / 'RESULT.json', dict(status='FINITE_BUDGET_FEASIBLE', scored_checkpoint='final.pt'))
            module.finalize = finalize
            return module
        modules = {'closure': types.ModuleType('closure'),
                   'closure.solver_binding': types.SimpleNamespace(adam400=adam400, restored_start=restore, oracle_at=Oracle,
                       isolated_solver=isolated, accepted=types.SimpleNamespace(old=types.SimpleNamespace(kkt=kkt))),
                   'pdf_information': types.SimpleNamespace(PDFProblem=ToyOverlay),
                   'development_information': types.SimpleNamespace(information=lambda p: None),
                   'score_forward': types.SimpleNamespace(score=lambda p,x,out: {'seconds':0.,'fixture':True})}
        identity = {'case': 'synthetic', 'strength': list(CASES.values())[0], 'nodes':80}
        original_save = core.save
        stopped = False
        def save_with_interrupt(path, obj):
            nonlocal stopped
            original_save(path, obj)
            if not stopped and path.parent.name == 'resumed':
                if interrupt == 'adam' and path.name == 'ADAM_RESUME.pt' and obj['completed_updates'] == 137:
                    run_fit.STOP = stopped = True
                if path.name == 'SOLVER_RESUME.pt':
                    if interrupt == 'solver' and obj['next_iteration'] == 2:
                        run_fit.STOP = stopped = True
                    if interrupt == 'terminal' and (obj['history'][-1].get('termination') or obj['next_iteration'] == 60):
                        run_fit.STOP = stopped = True
        with patch.dict(sys.modules, modules), patch.object(run_fit, 'binding', lambda case: identity), \
             patch.object(run_fit, 'configuration', lambda: {'backend':'cpu_guarded','threads':1,'solver_blas_threads':8}), \
             patch.object(core, 'problem', lambda task: (ToyProblem(), None)), \
             patch.object(core, 'check_scientific_bindings', lambda: None), \
             patch.object(core, 'environment', lambda: {'test': 'synthetic'}), \
             patch.object(core, 'load_module', loader), patch.object(core, 'save', save_with_interrupt):
            case = next(iter(CASES))
            a, b = self.root/'whole', self.root/'resumed'
            preflight=self.root/'preflight';preflight.mkdir(exist_ok=True)
            (preflight/'QUALIFICATION.json').write_text('{"fixture":true,"status":"PASS"}\n')
            self.assertEqual(run_fit.run(case, a), 0)
            self.assertEqual(run_fit.run(case, b), 75)
            self.assertTrue(stopped)
            run_fit.STOP = False
            self.assertEqual(run_fit.run(case, b, resume=True), 0)
            x = torch.load(a/'final.pt', weights_only=True)
            y = torch.load(b/'final.pt', weights_only=True)
            self.assertTrue(torch.equal(x['flat'], y['flat']))
            self.assertEqual(x['J'], y['J'])
            self.assertEqual(json.loads((a/'SOLVER_DIAGNOSTICS.json').read_text())['counts'],
                             json.loads((b/'SOLVER_DIAGNOSTICS.json').read_text())['counts'])
            self.assertEqual(json.loads((a/'SOLVER_DIAGNOSTICS.json').read_text())['oracle_counts'],
                             json.loads((b/'SOLVER_DIAGNOSTICS.json').read_text())['oracle_counts'])
            self.assertEqual(json.loads((b/'RESULT.json').read_text())['named_components']['J_joint'], y['J'])
            self.assertEqual((a/'preflight/QUALIFICATION.json').read_bytes(),(b/'preflight/QUALIFICATION.json').read_bytes())
            ca=json.loads((a/'OPTIMIZER_COMPLETE.json').read_text())['coordinate_cache_stats']
            cb=json.loads((b/'OPTIMIZER_COMPLETE.json').read_text())['coordinate_cache_stats']
            self.assertEqual({k:ca[k] for k in ('calls','hits','misses')},{k:cb[k] for k in ('calls','hits','misses')})
            with self.assertRaises(RuntimeError):
                run_fit.run(case, b, resume=True)
            with patch.object(run_fit, 'binding', lambda case: dict(identity, strength=99)):
                with self.assertRaises(RuntimeError):
                    run_fit.run(case, b, resume=True)

if __name__ == '__main__':
    unittest.main(verbosity=2)
