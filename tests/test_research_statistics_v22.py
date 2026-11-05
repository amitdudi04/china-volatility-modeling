"""Synthetic moving-block QLIKE comparison tests."""
import unittest
import numpy as np
import pandas as pd
from src.research_statistics_v22 import bootstrap_qlike_differences

class BootstrapTests(unittest.TestCase):
    @staticmethod
    def sample():
        t=np.arange(100)
        truth=np.where(t%2==0,.08,.16)
        return pd.DataFrame({
            "realized_next_variance":truth,
            "rolling_21_variance":np.full(100,.3),
            "better_variance":truth+.01,
            "equal_variance":np.full(100,.3),
        })

    def test_reproducible_bootstrap_intervals(self):
        f=self.sample()
        a=bootstrap_qlike_differences(f,blocks=5,replications=100,seed=18)
        b=bootstrap_qlike_differences(f,blocks=5,replications=100,seed=18)
        pd.testing.assert_frame_equal(a,b)
        better=a.loc[a.model=="better_variance"].iloc[0]
        equal=a.loc[a.model=="equal_variance"].iloc[0]
        self.assertLess(better.mean_qlike_difference,0)
        self.assertLess(better.ci95_high,0)
        self.assertAlmostEqual(equal.mean_qlike_difference,0)
        self.assertFalse(equal.ci_excludes_zero)

    def test_refuses_tiny_sample(self):
        with self.assertRaisesRegex(ValueError,"Insufficient"):
            bootstrap_qlike_differences(self.sample().iloc[:10])

if __name__=="__main__":
    unittest.main()
