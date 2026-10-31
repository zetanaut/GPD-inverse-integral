"""Isolated execution against the worker's existing, frozen final-geometry runtime.
No kernels or noise are generated. No production source or historical output is edited.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys,time,zipfile,platform,resource
from pathlib import Path
from study_core import file_sha, weighted_problem, training_prefix, continuous_summary, covariance_error_budget

def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)

def read(path):return json.loads(Path(path).read_text())

def safe_extract(archive,destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist())>100*1024**2:raise ValueError('Not a compact saved-fit archive')
        for i in z.infolist():
            if i.is_dir():continue
            dest=(destination/i.filename).resolve()
            if not dest.is_relative_to(destination) or ((i.external_attr>>16)&0o170000)==0o120000:raise ValueError('Unsafe archive path')
            raw=z.read(i)
            if dest.exists():
                if dest.read_bytes()!=raw:raise ValueError('Changed saved evidence: '+i.filename)
            else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)

class Runtime:
    def __init__(self,project):
        self.root=Path(project).resolve();self.prep=self.root/'completion_runs/NLO_final_geometry'
        # Original modules remain immutable. Evaluate only supplied sources, never an arbitrary fallback.
        lock=read(self.prep/'ESTIMATOR_LOCK.json')
        for name,h in lock['source_hashes'].items():
            if file_sha(self.root/name)!=h:raise ValueError('Original source lock mismatch: '+name)
        package_lock=read(Path(__file__).with_name('SOURCE_CONTRACT.json'))
        for name,h in package_lock['required_original_files'].items():
            if file_sha(self.root/name)!=h:raise ValueError('Audited source changed: '+name)
        os.environ['GPD_REVIEW_ROOT']=str(self.root)
        sys.path.insert(0,str(self.prep/'source'))
        import fit_final_geometry as fg
        self.fg=fg;self.a=fg.a;self.np=fg.np;self.torch=fg.torch
        self.torch.set_default_dtype(self.torch.float64);self.torch.set_num_threads(1);self.torch.use_deterministic_algorithms(True)
        self.bank=fg.load_bank()
        path=self.prep/'inputs/contractions.pt'
        if not path.exists():raise FileNotFoundError('Existing final contractions missing; do NOT rebuild: '+str(path))
        data=self.torch.load(path,weights_only=True)
        self.domain=self.a.Domain(data['points'],data['spin2'])
        if len(self.domain.names)!=265:raise ValueError('Wrong geometry/constraint contract')
        self.points=read(self.prep/'inputs/GEOMETRY_POINTS.json')
        self.binding={'estimator_lock_sha256':file_sha(self.prep/'ESTIMATOR_LOCK.json'),
          'geometry_sha256':file_sha(self.prep/'FINAL_GEOMETRY.json'),
          'contractions_sha256':file_sha(path),'fit_bank_sha256':file_sha(self.prep/'inputs/fit_bank.npz'),
          'study_source_sha256':file_sha(__file__),'study_core_sha256':file_sha(Path(__file__).with_name('study_core.py')),
          'python_executable':sys.executable,'python':sys.version,'platform':platform.platform(),
          'numpy':self.np.__version__,'torch':self.torch.__version__}
    def input(self,path):
        with self.np.load(path,allow_pickle=False) as z:
            if set(z.files)!={'y','covariance','row_ids'}:raise ValueError('Fitter input must contain ONLY observations/covariance/IDs')
            return z['y'].copy(),z['covariance'].copy(),z['row_ids'].tolist()
    def problem(self,path,seed,lam):
        y,C,ids=self.input(path)
        return weighted_problem(self.a.Problem,lam)(y,C,seed,self.bank,ids)
    def optimize(self,input_path,out,lam):
        out=Path(out);meta={**self.binding,'input_sha256':file_sha(input_path),'lambda_poly':float(lam)}
        marker=out/'OPTIMIZATION_COMPLETE_STUDY.json'
        if marker.exists():
            previous=read(marker)
            if previous['binding']!=meta:raise ValueError('Cannot reuse a changed study run')
            for name,h in previous['files'].items():
                if file_sha(out/name)!=h:raise ValueError('Saved optimization changed')
            return previous
        if (out/'STUDY_STARTED.json').exists():
            raise RuntimeError('Interrupted optimization retained; no automatic repeat or partial-fit selection')
        write(out/'STUDY_STARTED.json',meta)
        y,C,ids=self.input(input_path)
        def factory(yy,cc,seed):return weighted_problem(self.a.Problem,lam)(yy,cc,seed,self.bank,ids)
        fn,prefix_hash=training_prefix(self.a.smoke.run,{'Problem':factory,'Domain':lambda:self.domain})
        result=fn(y,C,out)
        files={str(p.relative_to(out)):file_sha(p) for p in sorted((out/'ranked').glob('candidate*.pt'))}
        files['CANDIDATE_ROSTER.json']=file_sha(out/'CANDIDATE_ROSTER.json')
        result.update(binding=meta,files=files,training_prefix_AST_sha256=prefix_hash)
        write(marker,result);return result
    def forward(self,p,x,out,binding):
        p.set_model(x);records=[];out=Path(out)
        for point in self.points:
            for q in point['Q2']:
                dest=out/f'xi{point["xi"]}_Q2{q}.json'
                if dest.exists():
                    wrapper=read(dest)
                    if wrapper['binding']!=binding:raise ValueError('Changed partial validation; refusing cache reuse')
                    result=wrapper['result']
                else:
                    result=self.fg.gv.pf.validate_point(point,q,'diagnostics_first_saved_or_weighted_fit',p.model)
                    write(dest,{'binding':binding,'result':result})
                row={k:result[k] for k in ('xi','Q2','pass','gates','CFF_estimated_total','CFF_limits')}
                row.update(field_totals={k:v['estimated_total'] for k,v in result['fields'].items()},geometry_role=point['role'],file=dest.name)
                records.append(row)
        return {'pass':all(r['pass'] for r in records),'records':records,'all_24_points':len(records)==24}
    def diagnose(self,input_path,fit_directory,out,lam):
        out=Path(out);fit=Path(fit_directory);roster=read(fit/'CANDIDATE_ROSTER.json')
        if not roster or roster!=sorted(roster,key=lambda c:(c['J'],c['seed'],c['stage'])):raise ValueError('Invalid original roster')
        results=[];start=time.monotonic()
        for i,c in enumerate(roster[:4]):
            path=fit/f'ranked/candidate{i}.pt';dest=out/f'candidate{i}'
            binding={**self.binding,'checkpoint_sha256':file_sha(path),'input_sha256':file_sha(input_path),'lambda_poly':float(lam)}
            saved=dest/'DIAGNOSTICS.json'
            if saved.exists():
                item=read(saved)
                if item['binding']!=binding:raise ValueError('Changed diagnostic cache')
            else:
                cp=self.torch.load(path,weights_only=True);p=self.problem(input_path,c['seed'],lam);x=cp['flat'].detach().clone()
                if (cp['seed'],cp['stage'])!=(c['seed'],c['stage']):raise ValueError('Checkpoint/roster mismatch')
                if not all(self.torch.equal(p.parameters(x)[k],v) for k,v in cp['state_dict'].items()):raise ValueError('State/flat mismatch')
                with self.torch.no_grad():
                    j=float(p.objective(x));orig=p.original_components(x).numpy()
                    if abs(j-c['J'])>1e-8*max(1,abs(j)):raise ValueError('Saved objective mismatch')
                    con=self.domain.constraints(p,x).numpy();m=self.a.smoke.moment_validation(p,x)
                    ta=self.a.smoke.targets(p,x,192);tb=self.a.smoke.targets(p,x,384)
                # This is the deliberate change: quality flags NEVER short-circuit full diagnostics.
                f=self.forward(p,x,dest/'full_forward',binding)
                quality=continuous_summary(m,f,con,float(max(abs(self.np.array(ta)-tb))))
                budgets={(float(r['xi']),float(r['Q2'])):r['CFF_estimated_total'] for r in f['records']}
                b=self.np.array([budgets[(float(v['xi']),float(v['Q2']))] for v in self.bank.points]).flatten()
                a=x.detach().requires_grad_(True);g=self.torch.autograd.grad(p.objective(a),a)[0]
                item={'binding':binding,'rank':i,'seed':c['seed'],'stage':c['stage'],'J_weighted':j,
                      'J_original_lambda1':float(orig.sum()),'data_chi2_per_scalar':float(orig[0]),'polynomiality_loss_unweighted':float(orig[1]),
                      'targets':tb,'raw_gradient_L2':float(g.norm()),'diagnostics':quality,
                      'numerics_in_noise_units':covariance_error_budget(b,p.cov.numpy()),
                      'full_forward':f,'moment_details':m,'checkpoint_path':str(path),'checkpoint_sha256':file_sha(path)}
                write(saved,item)
            results.append(item)
            # Fixed minimum-J order; no truth input and no cherry-picking a better-looking residual.
            if item['diagnostics']['computationally_evaluable']:
                selected=item;break
        else:selected=None
        result={'status':'finite_approximation_with_diagnostics' if selected else 'abstain_nonfinite',
                'selected':selected,'candidates_checked':results,'seconds_diagnostics':time.monotonic()-start,
                'phase':'development','selection':'lowest J/seed/stage computationally evaluable from original max-four roster; quality flags retained',
                'no_calibration':True,'historical_results_unchanged':True}
        write(out/'STUDY_RESULT.json',result);return result
