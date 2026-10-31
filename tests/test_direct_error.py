import unittest,math,dataclasses,json
from gpd_tools.direct_error import Contract,Fit,Pair,Cutoff,rank,joint_score,freeze_cutoff,intervals,evaluate,wilson

class Tests(unittest.TestCase):
 def setUp(self):self.c=Contract('model','physics',calibration_count=9)
 def pair(self,i,s='A',score=0.,phase='calibration',failure=None):
  f=Fit(f'{phase}_{s}_{i}',f'hash_{phase}_{s}_{i}',self.c.hash,None if failure else (score*.05,0.),failure)
  return Pair(f,s,(0.,0.),phase)
 def pairs(self):return [self.pair(i,s,float(i)/10) for s in self.c.strata for i in range(9)]
 def cutoff(self):return freeze_cutoff(self.c,self.pairs())
 def test_rank80(self):self.assertEqual(rank(80,'.10'),73)
 def test_rank9(self):self.assertEqual(rank(9,'.10'),9)
 def test_rank8(self):self.assertEqual(rank(8,'.10'),9)
 def test_scales(self):
  for v in (0.,-1.,float('nan'),float('inf')):
   with self.assertRaises(ValueError):Contract('m','d',scales=(v,.05))
 def test_bad_alpha(self):
  for v in ('0','1','-0.1','1.1'):
   with self.assertRaises(ValueError):Contract('m','d',alpha=v)
 def test_targets_unique(self):
  with self.assertRaises(ValueError):Contract('m','d',targets=('a','a'))
 def test_missing_binding(self):
  with self.assertRaises(ValueError):Contract('','d')
 def test_joint_max(self):
  p=dataclasses.replace(self.pair(1),fit=Fit('i','h',self.c.hash,(.05,-.10)))
  self.assertAlmostEqual(joint_score(self.c,p),2)
 def test_failure_score(self):self.assertTrue(math.isinf(joint_score(self.c,self.pair(0,failure='no feasible candidate'))))
 def test_nan_fit_is_not_failure(self):
  with self.assertRaises(ValueError):joint_score(self.c,dataclasses.replace(self.pair(0),fit=Fit('i','h',self.c.hash,(float('nan'),0))))
 def test_invalid_truth(self):
  with self.assertRaises(ValueError):joint_score(self.c,dataclasses.replace(self.pair(0),truth=(float('nan'),0)))
 def test_failed_finite_fit(self):
  with self.assertRaises(ValueError):Fit('i','h',self.c.hash,(0.,0.),'failed').checked(self.c)
 def test_finite_missing_values(self):
  with self.assertRaises(ValueError):Fit('i','h',self.c.hash,None).checked(self.c)
 def test_binding(self):
  with self.assertRaises(ValueError):Fit('i','h','wrong',(0.,0.)).checked(self.c)
 def test_missing_rows(self):
  with self.assertRaises(ValueError):freeze_cutoff(self.c,self.pairs()[:-1])
 def test_duplicates(self):
  p=self.pairs();p[-1]=p[-2]
  with self.assertRaises(ValueError):freeze_cutoff(self.c,p)
 def test_duplicate_input(self):
  p=self.pairs();p[-1]=dataclasses.replace(p[-1],fit=dataclasses.replace(p[-1].fit,input_hash=p[-2].fit.input_hash))
  with self.assertRaises(ValueError):freeze_cutoff(self.c,p)
 def test_dev_not_calibration(self):
  p=self.pairs();p[-1]=dataclasses.replace(p[-1],phase='development')
  with self.assertRaises(ValueError):freeze_cutoff(self.c,p)
 def test_unknown_stratum(self):
  p=self.pairs();p[-1]=dataclasses.replace(p[-1],stratum='C')
  with self.assertRaises(ValueError):freeze_cutoff(self.c,p)
 def test_exact_order_stat(self):self.assertAlmostEqual(self.cutoff().q,.8)
 def test_reordering(self):self.assertEqual(self.cutoff(),freeze_cutoff(self.c,list(reversed(self.pairs()))))
 def test_common_max(self):
  p=self.pairs();p[-1]=self.pair(8,'B',score=2.)
  q=freeze_cutoff(self.c,p);self.assertEqual(q.q,2.)
 def test_any_failure_infinite(self):
  p=self.pairs();p[0]=self.pair(0,failure='numerical')
  q=freeze_cutoff(self.c,p);self.assertIsNone(q.q);self.assertEqual(q.per_stratum['A']['failures'],1)
 def test_insufficient_sample_infinite(self):
  c=dataclasses.replace(self.c,calibration_count=8)
  p=[dataclasses.replace(self.pair(i,s),fit=dataclasses.replace(self.pair(i,s).fit,contract_hash=c.hash)) for s in c.strata for i in range(8)]
  self.assertIsNone(freeze_cutoff(c,p).q)
 def test_interval_truth_blind(self):
  out=intervals(self.c,self.cutoff(),self.pair(0,phase='test').fit)
  self.assertEqual(out['status'],'finite');self.assertAlmostEqual(out['bounds']['charge_bin_mean'][1],.04)
 def test_failed_test_full_line(self):
  out=intervals(self.c,self.cutoff(),self.pair(0,phase='test',failure='time budget').fit)
  self.assertEqual(out['status'],'fit_failure');self.assertTrue(all(v is None for v in out['bounds'].values()))
 def test_calibration_reuse(self):
  with self.assertRaises(ValueError):intervals(self.c,self.cutoff(),self.pair(0).fit)
 def test_changed_scale_invalidates(self):
  c=dataclasses.replace(self.c,scales=(.1,.05));f=dataclasses.replace(self.pair(0,phase='test').fit,contract_hash=c.hash)
  with self.assertRaises(ValueError):intervals(c,self.cutoff(),f)
 def test_json_infinities_are_explicit(self):
  p=self.pairs();p[0]=self.pair(0,failure='failed')
  q=freeze_cutoff(self.c,p);json.dumps(dataclasses.asdict(q),allow_nan=False)
 def test_all_term_selection_only(self):
  c=dataclasses.replace(self.c,estimator_hash='symbolically changed')
  self.assertNotEqual(c.hash,self.c.hash)
 def test_coverage_evaluation(self):
  p=[self.pair(i,s,score=0.,phase='test') for s in self.c.strata for i in range(40)]
  out=evaluate(self.c,self.cutoff(),p,expected_per_stratum=40)
  self.assertTrue(out['pass']);self.assertEqual(out['strata']['A']['joint_covered'],40)
 def test_failure_counts_not_dropped(self):
  p=[self.pair(i,s,phase='test',failure='failed' if i<3 else None) for s in self.c.strata for i in range(40)]
  out=evaluate(self.c,self.cutoff(),p,expected_per_stratum=40)
  self.assertFalse(out['pass']);self.assertEqual(out['strata']['A']['failed_fits'],3);self.assertEqual(out['strata']['A']['joint_covered'],40)
 def test_infinite_is_not_useful_success(self):
  p=self.pairs();p[0]=self.pair(0,failure='failed');q=freeze_cutoff(self.c,p)
  t=[self.pair(i,s,phase='test') for s in self.c.strata for i in range(40)]
  out=evaluate(self.c,q,t,expected_per_stratum=40);self.assertFalse(out['pass']);self.assertEqual(out['strata']['A']['finite_outputs'],0)
 def test_test_count_frozen(self):
  with self.assertRaises(ValueError):evaluate(self.c,self.cutoff(),[self.pair(0,phase='test')])
 def test_test_not_development(self):
  with self.assertRaises(ValueError):evaluate(self.c,self.cutoff(),[self.pair(0,phase='development')])
 def test_wilson(self):
  a,b=wilson(90,100);self.assertLess(a,.9);self.assertGreater(b,.9)
 def test_wilson_all(self):self.assertAlmostEqual(wilson(100,100)[1],1.)
 def test_wilson_bad(self):
  with self.assertRaises(ValueError):wilson(1,0)
 def test_no_profile_search_api(self):
  import inspect
  self.assertEqual(tuple(inspect.signature(intervals).parameters),('c','q','fit'))
 def test_finite_truth_bounds(self):
  p=self.pair(0,phase='test',score=1.)
  out=intervals(self.c,self.cutoff(),p.fit);self.assertGreater(out['bounds']['charge_bin_mean'][0],0.)

if __name__=='__main__':unittest.main(verbosity=2)
