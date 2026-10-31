"""Fail before fitting if the pinned local execution environment has changed."""
from pathlib import Path
import sys,json,platform,os
PREP=Path(__file__).resolve().parents[1]

def verify():
 import numpy,scipy,torch
 from threadpoolctl import threadpool_info
 expected=json.loads((PREP/'runtime_repair/ENVIRONMENT.json').read_text())
 actual={'sys_executable':sys.executable,'resolved_executable':str(Path(sys.executable).resolve()),
         'python':sys.version,'platform':platform.platform(),'numpy':numpy.__version__,
         'scipy':scipy.__version__,'torch':torch.__version__,
         'thread_settings':{k:os.environ.get(k) for k in expected['thread_settings']},
         'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads()}
 for key,value in actual.items():
  if value!=expected[key]:raise RuntimeError('Pinned environment mismatch: '+key)
 # Imported BLAS/OpenMP implementations may include SciPy's separate BLAS copy.
 # Require each loaded pool to use one thread; the exact list is recorded.
 actual['loaded_threadpools']=threadpool_info()
 if any(pool['num_threads']!=1 for pool in actual['loaded_threadpools']):
  raise RuntimeError('A numerical runtime has more than one thread')
 return actual
