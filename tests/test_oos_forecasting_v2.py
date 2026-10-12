"""Causal forecasting regression tests; all use synthetic prices."""
import unittest
import numpy as np
import pandas as pd
from src.research_design import prepare_session_dataset
from src.oos_forecasting_v2 import (
    forecast_one_origin, walk_forward, score_variance_predictions,
)

class ForecastingTests(unittest.TestCase):
    @staticmethod
    def dataset():
        idx=pd.bdate_range("2020-01-01", periods=370)
        step=np.arange(len(idx),dtype=float)
        close=100*np.exp(np.cumsum(.0003+.012*np.sin(step*.43)))
        return prepare_session_dataset(pd.DataFrame({"Close":close},index=idx))

    def test_forecast_causal_training_maturity(self):
        f=self.dataset()
        t=f.index[300]
        x=forecast_one_origin(f,t,min_training=100,
                              include_garch=False,n_estimators=8)
        self.assertLess(x["training_observations"], 300)
        self.assertGreater(x["training_observations"], 100)
        self.assertAlmostEqual(
            x["har_rf_equal_variance"],
            (x["har_ridge_variance"]+x["random_forest_variance"])/2)
        self.assertTrue(x["har_rf_equal_variance"]>0)

    def test_future_changes_do_not_change_prior_prediction(self):
        f=self.dataset()
        t=f.index[301]
        a=forecast_one_origin(f,t,min_training=100,
                              include_garch=False,n_estimators=8)
        b=f.copy()
        b.loc[b.index>b.index[310],"target_variance_ann"]=123.45
        q=forecast_one_origin(b,t,min_training=100,
                              include_garch=False,n_estimators=8)
        for key in ("rolling_21_variance","har_ridge_variance",
                    "random_forest_variance","har_rf_equal_variance"):
            self.assertAlmostEqual(a[key],q[key])

    def test_walk_forward_alignment_and_metrics(self):
        f=self.dataset()
        p=walk_forward(f,str(f.index[300].date()),
                       str(f.index[307].date()),
                       min_training=100,max_origins=3,
                       include_garch=False,n_estimators=8)
        self.assertEqual(len(p),3)
        self.assertTrue((pd.to_datetime(p["target_end_date"])>p.index).all())
        m=score_variance_predictions(p)
        self.assertEqual(set(m["model"]),
                         {"rolling_21_variance","har_ridge_variance",
                          "random_forest_variance","har_rf_equal_variance"})
        self.assertTrue(np.isfinite(m[["rmse","mae","qlike"]]).all().all())

    def test_short_history_fails(self):
        f=self.dataset()
        with self.assertRaisesRegex(ValueError,"Insufficient"):
            forecast_one_origin(f,f.index[10],include_garch=False)

    def test_garch_produces_real_conditional_variance(self):
        try:
            import arch
        except ImportError:
            self.skipTest("arch is not installed in minimalist test runner")
        f=self.dataset()
        p=forecast_one_origin(f,f.index[300],
                              min_training=100,include_garch=True,
                              n_estimators=8)
        self.assertTrue(np.isfinite(p["garch11_variance"]))
        self.assertGreater(p["garch11_variance"],0)

if __name__=="__main__":
    unittest.main()
