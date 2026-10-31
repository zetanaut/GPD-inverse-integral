from pathlib import Path
import sys,json,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candidate'))
from forward import load,initial,hard_row
from src.ordered_evolution import evolve
ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--q2',type=float,default=2.3);a=ap.parse_args();b,k0,k1,key=load(a.directory);v=initial(b);h,diag=evolve(v,k0,k1,a.q2,rtol=1e-11,atol=1e-13);np.savez_compressed(a.directory/'smooth_field.npz',initial=v,evolved=h,nodes=b.nodes);row,hd=hard_row(b,a.q2,96);c=row@h;(a.directory/'SOLVE.json').write_text(json.dumps({'CFF':[c.real,c.imag],'q2':a.q2,'evolution':diag,'hard':hd},indent=2)+'\n');print(c)
