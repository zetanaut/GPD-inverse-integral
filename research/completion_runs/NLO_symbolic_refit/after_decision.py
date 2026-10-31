"""Truth diagnostics and manuscript fragment, only after decisions are frozen."""
from driver import *
from fractions import Fraction as F

def exact_true_coefficients(truth,channel,spin):
    # Frozen structural development functions, integrated analytically by parity.
    # Full-support int x^N(1-x^2)^2 = 16/[(N+1)(N+3)(N+5)].
    if channel<2:scale=F(16,(spin+1)*(spin+3)*(spin+5))
    else:scale=F(16,(spin-1)*(spin+1)*(spin+3))
    amplitude=[F(2),F(1,2),F(4,5)][channel]
    xi_factor=[F(1,5),F(-1,10),F(1,10)][channel]
    coeff=[F(0)]*(spin//2+1);coeff[0]=amplitude*scale;coeff[1]=amplitude*xi_factor*scale
    if truth=='Dterm_added':coeff[spin//2]+=[F(9,100),F(-1,50),F(3,100)][channel]*scale
    return coeff

def expression(values,supports,variable):
    # Round-trip binary64 decimal display, not a new coefficient constraint.
    terms=[]
    for k in supports:
        power=k if variable=='z' else 2*k
        coef=format(float(values[k]),'.17g')
        terms.append(coef if power==0 else f'({coef})*{variable}' if power==1 else f'({coef})*{variable}^{power}')
    return ' + '.join(terms) if terms else '0'

def free_templates(channel,spin,support):
    symbols={k:f'a[{channel},{spin},{k}]' for k in support}
    z=' + '.join(symbols[k] if k==0 else f'{symbols[k]}*z^{k}' for k in support) or '0'
    xi=' + '.join(symbols[k] if k==0 else f'(25/9)^{k}*{symbols[k]}*xi^{2*k}' for k in support) or '0'
    return {'free_coefficient_template_z':z,'free_coefficient_template_xi':xi}

def checkpoint_path(result):
    key=result.get('reused_branch',result['key'])
    return ROOT/'branches'/key/'final.pt'

def main():
    decision_path=ROOT/'DECISIONS.json';assert decision_path.exists(),'No truth diagnostics before frozen decisions'
    decision_hash=c.digest(decision_path);decisions=json.loads(decision_path.read_text())
    props=json.loads((ROOT/'PROPOSALS.json').read_text());summary=json.loads((ROOT/'BRANCH_SUMMARY.json').read_text())
    parents={v['key']:v for v in PLAN['parents']};byprop={v['parent']:v for v in props['parents']}
    diagnostics=[];false_simplifications=[];expressions=[]
    for result in summary['branches']:
        parent=parents[result['parent']];truth=parent['truth']
        saved=torch.load(checkpoint_path(result),weights_only=True)
        model=c.ReferenceNetwork('film').double();model.load_state_dict(saved['state_dict'])
        field=n.truth_diagnostics(model,truth)
        coefficient_checks=[]
        entries=result['full_space']['orders']['192']['entries']
        selected=result['selected_space']['orders']['192']['entries']
        for index,(full,chosen) in enumerate(zip(entries,selected)):
            ch=index//4;spin=full['spin'];true=exact_true_coefficients(truth,ch,spin)
            tx=np.array(list(map(float,true)));tz=tx*.6**(2*np.arange(len(tx)))
            full_x=np.array(full['coefficient_xi']);chosen_x=np.array(chosen['coefficient_xi'])
            scale=PLAN['moments']['scales'][index]
            coefficient_checks.append({'channel':CHANNELS[ch],'spin':spin,
               'true_coefficients_xi_exact_rational':[str(v) for v in true],
               'true_coefficients_xi':tx.tolist(),'true_coefficients_z':tz.tolist(),
               'full_projection_coefficient_xi':full_x.tolist(),'declared_subspace_coefficient_xi':chosen_x.tolist(),
               'full_projection_xi_coefficient_error':(full_x-tx).tolist(),
               'declared_subspace_xi_coefficient_error':(chosen_x-tx).tolist(),
               'full_projection_z_coefficient_error_over_moment_scale':((np.array(full['coefficient_z'])-tz)/scale).tolist()})
            expressions.append({'branch':result['key'],'channel':CHANNELS[ch],'spin':spin,
               'role':'post-refit inferred moments; descriptive coefficients, not fixed in loss',
               'support_z':chosen['support_z'],'powers_xi':chosen['powers_xi'],
               'coefficient_z':chosen['coefficient_z'],'coefficient_xi':chosen['coefficient_xi'],
               'expression_z':expression(chosen['coefficient_z'],chosen['support_z'],'z'),
               'expression_xi':expression(chosen['coefficient_xi'],chosen['support_z'],'xi')})
        diagnostics.append({'key':result['key'],'field_recovery':field,'coefficient_checks':coefficient_checks})
    for proposal in props['parents']:
        parent=parents[proposal['parent']];decision=next(d for d in decisions['decisions'] if d['parent']==parent['key'])
        records=[]
        for index,rec in enumerate(proposal['records']):
            true=exact_true_coefficients(parent['truth'],rec['channel_index'],rec['spin'])
            omitted=[k for k,v in enumerate(true) if v!=0 and k not in rec['support']]
            extra=[k for k in rec['support'] if true[k]==0]
            records.append({'channel':CHANNELS[rec['channel_index']],'spin':rec['spin'],'support_z':rec['support'],
              'true_nonzero_support_z':[k for k,v in enumerate(true) if v!=0],
              'false_zero_proposals':[{'z_power':k,'xi_power':2*k,'true_xi_coefficient_exact':str(true[k])} for k in omitted],
              'retained_true_zero_powers_z':extra})
            expressions.append({'branch':parent['key']+'__proposal','channel':CHANNELS[rec['channel_index']],'spin':rec['spin'],
              'role':'teacher proposal only; never fixed in optimization','support_z':rec['support'],'powers_xi':rec['xi_powers'],
              'coefficient_z':rec['coefficient_scaled'],'coefficient_xi':rec['coefficient_xi'],
              'expression_z':expression(rec['coefficient_scaled'],rec['support'],'z'),
              'expression_xi':expression(rec['coefficient_xi'],rec['support'],'xi')})
        false_simplifications.append({'parent':parent['key'],'candidate_compatible':decision['compatible'],'moments':records,
            'false_zero_count':sum(len(r['false_zero_proposals']) for r in records)})
    analytic_checks=[]
    for truth_name in ('smooth','Dterm_added'):
        projector=c.projector(192);x=projector.x.numpy();xi=projector.xi.numpy()
        h=np.stack([n.truth_values(xx,ss,truth_name=='Dterm_added') for xx,ss in zip(x,xi)])
        actual=np.column_stack([(h[:,ch]*getattr(projector,f'xw{i}').numpy()).sum(-1) for i,(ch,spin,power) in enumerate(projector.spec)])
        predicted=np.column_stack([c.poly.design(xi,spin)@(np.array(list(map(float,exact_true_coefficients(truth_name,ch,spin))))*.6**(2*np.arange(spin//2+1)))
                                   for ch,spin,power in projector.spec])
        err=float(np.max(abs(actual-predicted)));assert err<=1e-12
        analytic_checks.append({'truth':truth_name,'quadrature_per_side':192,'max_absolute_moment_identity_error':err,'pass':True})
    after={'created_UTC':timestamp(),'decisions_sha256_before_truth_evaluation':decision_hash,'diagnostics':diagnostics,
      'analytic_truth_coefficient_checks':analytic_checks,
      'false_simplifications':false_simplifications,'meaning':'truth diagnostics AFTER frozen decisions; no tuning or acceptance feedback'}
    write(ROOT/'AFTER_DECISION_TRUTH.json',after)
    for e in expressions:e.update(free_templates(e['channel'],e['spin'],e['support_z']))
    write(ROOT/'EXPRESSIONS.json',{'coordinate':'z=(xi/0.6)^2; c_xi[2k]=c_z[k]/0.6^(2k)',
      'display':'17-digit round-trip values; complete coefficient arrays supplied; symbolic supports are exact, estimated coefficients are not exact physical constants',
      'expressions':expressions})
    lines=['# Symbolic moment expressions','',
      'Here z=(xi/0.6)^2. Listed k values are powers of z; the corresponding xi powers are 2k. Coefficients are inferred, remain free during refitting, and are printed at round-trip precision. No coefficient is an exact physical constant.','',
      '| Branch | Moment | z support | Expression in z | Expression in xi |','|---|---|---|---|---|']
    for e in expressions:lines.append(f"| {e['branch']} | {e['channel']}, N={e['spin']} | {e['support_z']} | `{e['expression_z']}` | `{e['expression_xi']}` |")
    (ROOT/'EXPRESSIONS.md').write_text('\n'.join(lines)+'\n')
    make_tables(summary,props,decisions,after)
    assert c.digest(decision_path)==decision_hash,'Decisions changed during truth reporting'

def make_tables(summary,props,decisions,truth):
    lines=['# Equal-budget all-term and symbolic-subset continuations','',
      'Each pair starts from the identical accepted NLO parent and has at most 60 Gauss–Newton Jacobians plus five cleanup steps. All-term remains nominal. Equal allowed budgets do not imply equal executed work.','',
      '| Branch | Terms | J | Fresh gradient | Full RMS (192) | Selected RMS (192) | GN updates / cleanup | Gates |',
      '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in summary['branches']:
        lines.append(f"| {r['key']} | {sum(map(len,r['supports_z']))} | {r['J']:.6g} | {r['gradient_l2']:.6g} | {r['full_space']['orders']['192']['combined_rms']:.6g} | {r['selected_space']['orders']['192']['combined_rms']:.6g} | {r['accepted_GN_updates']} / {r['accepted_cleanup_updates']} | {'pass' if r['pass'] else 'FAIL: '+', '.join(k for k,v in r['gates'].items() if not v)} |")
    lines+=['','## Frozen candidate decisions','',
      '| Parent | Terms: full → proposal | Both branches pass | Selection rule | Decision | After-decision false-zero count |',
      '|---|---:|---|---|---|---:|']
    for d,p,t in zip(decisions['decisions'],props['parents'],truth['false_simplifications']):
        selected=all(all(x['component_pass']) for x in d['selection_components'])
        lines.append(f"| {d['parent']} | {p['terms_full']} → {p['terms_proposed']} | {d['allterm_pass'] and d['candidate_pass']} | {selected} | {d['interpretation']} | {t['false_zero_count']} |")
    lines+=['','## Residuals and after-decision recovery','',
       'Maximum absolute Re/Im residual per subset; recovery is maximum relative x-integrated L2 over the three existing xi slices. Q²=6.1 and truth recovery do not enter the selection inequality.','',
       '| Branch | Training Re / Im | Selection Q²=3.1 Re / Im | Reporting Q²=6.1 Re / Im | Charge recovery | Sigma / T / Hg recovery |',
       '|---|---:|---:|---:|---:|---:|']
    fmt=lambda vals:' / '.join(f'{v:.5g}' for v in vals)
    for r,t in zip(summary['branches'],truth['diagnostics']):
        values=[np.max(np.abs([x['reference_residual'] for x in r['CFF'] if x['split']==split]),axis=0) for split in ('training','selection','reporting')]
        field=t['field_recovery']['points'];charge=max(e['charge_relative_L2'] for e in field)
        components=np.max([e['component_relative_L2'] for e in field],axis=0)
        lines.append(f"| {r['key']} | {fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])} | {charge:.5%} | {fmt(100*components)} % |")
    lines+=['','Numerical budgets and all six selection inequalities per parent are retained in the JSON records. Apparent residual precision below those empirical budgets is not physical accuracy. False-zero counts are after-decision diagnostics, not selection criteria.','']
    (ROOT/'COMPARISON.md').write_text('\n'.join(lines))
    compatible=sum(d['compatible'] for d in decisions['decisions'])
    false_total=sum(t['false_zero_count'] for t in truth['false_simplifications'])
    fragment=f'''## Methods: conditional symbolic moment hypotheses

The nominal reconstruction was the original reference-scale FiLM neural GPD with
all allowed even-power moment coefficients free at spins N=2,4,6,8. The accepted
fixed-nf=3, MSbar NLO forward map, two-loop running, data normalization 0.04 and
moment penalty weight one were retained. No Q² input was added to the network.
For each of four accepted NLO development fits, one deterministic symbolic support
proposal approximated the full polynomial projection of its inferred moments at
relative tolerance 0.001. Supports are powers k of z=(xi/0.6)^2, corresponding to
xi powers 2k. Proposal coefficient estimates were recorded but never fixed in the
refit; all retained coefficients remained variable-projected unknowns.

Each proposed support and an all-term control started from the same fitted parent,
with a common allowed continuation budget of 60 Gauss–Newton Jacobians and at most
five cleanup steps. Fresh stationarity, full/selected moment-space consistency,
96-versus-192-point moment quadrature, and the accepted NLO forward numerical
checks were required. Candidate compatibility used only the previously inspected
Q²=3.1 development subset, at each complex-CFF component, with 1.10 times the
all-term residual plus the two empirical numerical allowances. Q²=6.1 was retained
for reporting; neither subset was a fresh final or unseen-skewness test.

## Results and scope

There were {summary['nominal_branches']} nominal branches and {summary['actual_continuations']} actual continuation calls.
{compatible} of four proposals satisfied the declared compatibility rule. This means
unresolved-compatible descriptive compression, not discovery of physical zeros.
After decisions were frozen, comparison with the structural development truths
identified {false_total} omitted nonzero moment coefficients across the four proposals;
all such false simplifications are retained explicitly. Numerical floors limit the
interpretation of very small CFF residuals. The all-term fit remains the nominal
estimator, and the conditional alternatives define no uncertainty model.

The preceding order-control study accepted four LO→LO and four NLO→NLO matched
fits; all four NLO→LO mismatch fits failed stationarity, polynomiality and learned-field
numerical gates. Their field shifts are not converged perturbative-bias estimates
and those failed fits were not restarted here. These simple, previously inspected
development examples establish neither universal inverse identifiability nor
calibrated nonlinear uncertainty or experimental validity. The next step is a frozen
final estimator/target/domain and calibration design, not an architecture survey.

Internal evidence: [order-comparison review](../../GPD_NLO_order_comparison_review/REPORT.md),
[frozen symbolic decisions](DECISIONS.json), and [paired branch results](COMPARISON.md).
'''
    (ROOT/'MANUSCRIPT_FRAGMENT.md').write_text(fragment)

if __name__=='__main__':main()
