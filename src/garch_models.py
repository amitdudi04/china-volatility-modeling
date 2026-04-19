from src.utils.validation import validate_and_save
import os
import numpy as np
import pandas as pd
from arch import arch_model
from src.utils import get_market_path

class GARCHModels:
    def estimate(self, returns, current_model, p=1, q=1):
        # STEP 1: FORCE SCALING AT MODEL INPUT
        returns_scaled = returns * 100
        
        # STEP 2: DISABLE ARCH WARNING
        if current_model == 'ARCH':
            # ARCH(1) is basically GARCH(p=1, q=0)
            am = arch_model(returns_scaled, vol='ARCH', p=p, dist='t', rescale=False)
        elif current_model == 'GARCH':
            am = arch_model(returns_scaled, vol='GARCH', p=p, q=q, dist='t', rescale=False)
        elif current_model == 'EGARCH':
            # EGARCH models require the o argument for asymmetric shocks. E.g. o=1
            am = arch_model(returns_scaled, vol='EGARCH', p=p, o=1, q=q, dist='t', rescale=False)
        elif current_model == 'GJR-GARCH':
            # GJR-GARCH uses o=1 with standard GARCH framework in arch package
            am = arch_model(returns_scaled, vol='GARCH', p=p, o=1, q=q, dist='t', rescale=False)
        else:
            raise ValueError("Unsupported Model")
            
        res = am.fit(disp='off')
        return res

    def compare_models(self, df, market_name):
        models = ['ARCH', 'GARCH', 'EGARCH', 'GJR-GARCH']
        returns = df['Log_Return'] # Decimal formatting formally required natively
        print(f"[DEBUG GARCH] Input length: {len(df)}")
        assert len(df) > 200, f"GARCH received too little data: {len(df)}"
        results = []
        
        print(f"Estimating Volatility Models for {market_name}...")
        for mod in models:
            try:
                res = self.estimate(returns, mod)
                results.append({
                    'Market': market_name,
                    'Model': mod,
                    'AIC': res.aic,
                    'BIC': res.bic,
                    'LogLikelihood': res.loglikelihood,
                    'Params': str(res.params.to_dict())
                })
            except Exception as e:
                print(f"Estimation failed for {mod} in {market_name}: {e}")
                
        metrics_df = pd.DataFrame(results).set_index("Model")
        validate_and_save(metrics_df, os.path.join(get_market_path(market_name), f"{market_name}_garch_comparison.csv"), is_time_series=False)
        print(metrics_df[['AIC', 'BIC']])
        return metrics_df
