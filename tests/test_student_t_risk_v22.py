"""Student-t standardized conditional VaR and hit-independence checks."""
import unittest
import numpy as np
from scipy.stats import norm, t
from src.student_t_risk_v22 import standardized_t_var_es,christoffersen_tests

class StudentTRiskTests(unittest.TestCase):
    def test_standardization_and_var_magnitude(self):
        a,es=standardized_t_var_es(.0002,.012,8,.05)
        z=t.ppf(.05,8)*np.sqrt(6/8)
        self.assertAlmostEqual(a,.0002+.012*z)
        self.assertLess(es,a)
        self.assertLess(a,.0002)
        self.assertTrue(np.isfinite(es))
    def test_very_large_nu_approaches_normal(self):
        a,_=standardized_t_var_es(0,.015,10000,.05)
        self.assertAlmostEqual(a,.015*norm.ppf(.05),places=5)
    def test_bad_inputs_rejected(self):
        for mu,sigma,nu in [(0,0,7),(0,.01,2),(0,.01,np.nan)]:
            with self.subTest(mu=mu,sigma=sigma,nu=nu):
                with self.assertRaises(ValueError):
                    standardized_t_var_es(mu,sigma,nu)
    def test_independence_and_conditional_coverage_boundaries(self):
        for hits in ([False]*75,[True]*75,[False,True]*40):
            stats=christoffersen_tests(hits)
            self.assertEqual(stats["n00"]+stats["n01"]+stats["n10"]+stats["n11"],len(hits)-1)
            self.assertTrue(0<=stats["christoffersen_ind_pvalue"]<=1)
            self.assertTrue(0<=stats["christoffersen_cc_pvalue"]<=1)
            self.assertTrue(np.isfinite(stats["christoffersen_ind_lr"]))
    def test_clustered_hits_more_dependent_than_spaced_hits(self):
        cluster=np.r_[np.zeros(60,dtype=int),np.ones(20,dtype=int),np.zeros(60,dtype=int)]
        spaced=np.array([1 if i%7==0 else 0 for i in range(140)])
        a=christoffersen_tests(cluster)
        b=christoffersen_tests(spaced)
        self.assertGreater(a["christoffersen_ind_lr"],b["christoffersen_ind_lr"])
if __name__=="__main__":
    unittest.main()
