"""Risk and economic accounting are tested on manually auditable prices."""
import unittest
import numpy as np
import pandas as pd
from src.risk_economic_v22 import (
    align_next_session_returns,kupiec_unconditional_coverage,
    normal_var_coverage,economic_exposure_sensitivity,
)

class RiskEconomicsV22Tests(unittest.TestCase):
    @staticmethod
    def fixture():
        dates=pd.bdate_range("2024-01-01",periods=5)
        raw=pd.DataFrame({"Close":[100.,110.,88.,96.,100.]},index=dates)
        pred=pd.DataFrame({
            "target_end_date":dates[1:],
            "garch11_variance":[.16,.16,.16,.16],
            "rolling_21_variance":[.09,.09,.09,.09],
            "realized_next_variance":[.1,.1,.1,.1],
        },index=dates[:-1])
        return raw,pred

    def test_exact_next_session_return_mapping(self):
        raw,pred=self.fixture()
        x=align_next_session_returns(pred,raw)
        np.testing.assert_allclose(x.to_numpy(),np.log(np.array([1.1,.8,96/88,100/96])))

    def test_wrong_target_session_is_rejected(self):
        raw,pred=self.fixture()
        pred.loc[pred.index[0],"target_end_date"]=raw.index[2]
        with self.assertRaisesRegex(ValueError,"exactly the next"):
            align_next_session_returns(pred,raw)

    def test_static_strategy_compounding_equals_price_return(self):
        raw,pred=self.fixture()
        x=align_next_session_returns(pred,raw)
        table=economic_exposure_sensitivity(pred,x,cost_bps=0,target_annual_vol=.2)
        static=table.set_index("strategy").loc["static"]
        managed=table.set_index("strategy").loc["volatility_managed"]
        self.assertAlmostEqual(static.cumulative_return,0.)
        self.assertAlmostEqual(static.average_exposure,1.)
        self.assertAlmostEqual(managed.average_exposure,.5)
        self.assertLess(static.max_drawdown,0)

    def test_nonzero_turnover_cost_reduces_strategy_return(self):
        raw,pred=self.fixture();x=align_next_session_returns(pred,raw)
        free=economic_exposure_sensitivity(pred,x,cost_bps=0)
        charged=economic_exposure_sensitivity(pred,x,cost_bps=20)
        f=free.set_index("strategy");c=charged.set_index("strategy")
        self.assertAlmostEqual(f.loc["static","cumulative_return"],c.loc["static","cumulative_return"])
        self.assertGreater(f.loc["volatility_managed","cumulative_return"],
                           c.loc["volatility_managed","cumulative_return"])

    def test_kupiec_handles_zero_and_all_breaches(self):
        for b in ([False]*50,[True]*50):
            x=kupiec_unconditional_coverage(b)
            self.assertTrue(np.isfinite(x["kupiec_lr"]))
            self.assertTrue(0<=x["kupiec_pvalue"]<=1)

    def test_normal_var_output_is_labelled_normal(self):
        raw,pred=self.fixture(); x=align_next_session_returns(pred,raw)
        v=normal_var_coverage(pred,x)
        self.assertEqual(set(v.model),{"rolling_21_variance","garch11_variance"})
        self.assertEqual(set(v["observations"]),{4})

if __name__=="__main__":
    unittest.main()
