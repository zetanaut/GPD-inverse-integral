"""Stage the explicit review list for portable replay and the supplied packager."""
from pathlib import Path
import shutil,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 result=json.loads((ROOT/'work/RESULT.json').read_text())
 if result['status']!='passed_for_scientific_review':raise RuntimeError('Milestone not passed')
 stage=ROOT/'work/review_stage';stage.mkdir(exist_ok=False);names=(ROOT/'work/REVIEW_FILES.txt').read_text().splitlines()
 for name in names:
  src=ROOT/name;dest=stage/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
  if hashlib.sha256(src.read_bytes()).digest()!=hashlib.sha256(dest.read_bytes()).digest():raise RuntimeError('Copy changed '+name)
 for name in ['SUMMARY.md','RESULT.json']:
  shutil.copy2(stage/'work'/name,stage/name);names.append(name)
 readme='''# Task 03 resolution review\n\nRead SUMMARY.md, RESULT.json, and candidate/FORWARD_API.json.\n\nFrom this extracted review directory, replay the six saved full-support coupled NLO fixtures with:\n\n```bash\nOMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python candidate/reproduce_review.py\n```\n\nThe tested replay environment is Python3.9.23, NumPy1.26.3 and SciPy1.13.0. Replay verifies the review hashes, recomputes coupled fields from the saved finite operators, checks physical-x field errors, independent continuum coefficient quadrature, and the adjoint map. It constructs no kernels and performs no fits. All numerical budgets and the lower-degree native fields are included. The original package's START_HERE/preflight directions refer to its larger immutable input inventory; this review uses the command above.\n\nSource arithmetic/build metadata are in work/action/MP_BUILD.json and FAST_QG_BUILD.json. Optional source regeneration requires the existing Fortran/C++ toolchain, MPFR/GMP and mpmath recorded there. The source files, prescribed guards, equation mapping, original-source hashes and candidate diff are included; platform-specific binaries and large row caches are omitted. Exact local paths and hashes identify retained caches in the original project. No dependency installation is automatic.\n\nThe evolved reference uses a distinct strong discretization of the pinned source. It shares source functions with the weak candidate; independent action/transcription, published moment, hard coefficient, ordering and derivative controls are documented explicitly. Validation is limited to the frozen development fixtures, rather than a claim about arbitrary input functions or experimental observables.\n'''
 (stage/'README.md').write_text(readme);names.append('README.md');(stage/'REVIEW_FILES.txt').write_text('\n'.join(sorted(set(names)))+'\n');print(stage)
if __name__=='__main__':main()
