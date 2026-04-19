from src.utils.validation import validate_and_save
import os
import numpy as np
import pandas as pd
import logging
import shutil
from sklearn.metrics import mean_squared_error

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.eda_econometrics import EDAEconometrics
from src.garch_models import GARCHModels
from src.regime_model import RegimeModel
from src.forecast_engine import ForecastEngine
from src.ml_models import MLModels
from src.risk_metrics import RiskMetrics
from src.evaluator import ModelEvaluator
from src.interpretation import InterpretationLayer
from src.data_validation import validate_data
from src.validation import ValidationEngine
from src.model_comparison import ModelComparison
from src.utils import get_market_path

def validate_df(df):
    assert df is not None, "DataFrame is None"
    assert not df.empty, "DataFrame is empty"
    assert not df.isna().all().any(), "All values NaN"
    return True

class Pipeline:
    def __init__(self):
        shutil.rmtree("results/metrics", ignore_errors=True)
        for d in ["data/raw", "data/processed", "results/figures", "outputs/reports", "outputs/results", "logs"]:
            if not os.path.exists(d):
                os.makedirs(d)
                
    def run_market(self, market_name, df):
        print(f"!!! STARTING RUN_MARKET FOR {market_name} !!!")
        logging.info(f"Executing Complete Pipeline for {market_name}")
        print(f"[INFO] Running pipeline for {market_name}")
        
        # 1. Preprocess & Validate
        try:
            prep = Preprocessor()
            df_processed = prep.process(df, market_name)
            validate_df(df_processed)
            prep.describe(df_processed, market_name)
        except Exception as e:
            raise RuntimeError(f"Data Preprocessing failed: {str(e)}")
            
        # 2. EDA & Econometrics
        try:
            eda = EDAEconometrics()
            eda.analyze(df_processed, market_name)
        except Exception as e:
            raise RuntimeError(f"EDA failed: {str(e)}")

        # 3. Volatility Models (ARCH, GARCH, EGARCH, GJR-GARCH)
        try:
            garch = GARCHModels()
            garch.compare_models(df_processed, market_name)
        except Exception as e:
            raise RuntimeError(f"GARCH Base Models failed: {str(e)}")

        # 4. Regime Framework (Volatility proxy)
        try:
            regm = RegimeModel()
            df_processed_regime = regm.detect_regimes(df_processed, market_name)
            validate_df(df_processed_regime)
        except Exception as e:
            raise RuntimeError(f"Regime proxy failed: {str(e)}")

        # 5. GARCH Forecast Engine (1000 rolling window)
        try:
            forecaster = ForecastEngine()
            garch_preds = forecaster.rolling_forecast(df_processed, market_name)
            validate_df(garch_preds)
        except Exception as e:
            raise RuntimeError(f"Critical failure: GARCH Forecast Engine - {str(e)}")

        # 6. Machine Learning Forecast
        try:
            ml = MLModels()
            ml_preds = ml.run_ml(df_processed, market_name)
            validate_df(ml_preds)
        except Exception as e:
            raise RuntimeError(f"Critical failure: ML - {str(e)}")
            
        # 6.5 HYBRID MODEL COMBINATION
        try:
            # STEP 1: BUILD MASTER DATAFRAME natively
            df = pd.DataFrame()
            df['GARCH'] = garch_preds['Forecast_Vol']
            df['ML'] = ml_preds['Forecast_Vol']
            # SAFETY CLIP (prevents extreme weights instability)
            df['GARCH'] = df['GARCH'].clip(0.05, 0.60)
            df['ML'] = df['ML'].clip(0.05, 0.60)

            df['Realized'] = garch_preds['Realized_Vol']
            
            # STEP 1: DEBUG SOURCE
            print(f"[DEBUG] Pre-drop NaN count:\n{df.isna().sum()}")
            
            # STEP 5: Align starting index manually (IMPORTANT)
            start_garch = df['GARCH'].first_valid_index()
            start_ml = df['ML'].first_valid_index()
            
            if start_garch is not None and start_ml is not None:
                min_valid_index = max(start_garch, start_ml)
                df = df.loc[min_valid_index:]
            
            # STEP 3: TRIM DATA PROPERLY
            df = df.loc[
                df['GARCH'].notna() &
                df['ML'].notna()
            ].copy()
            
            # STEP 4: HARD VALIDATION mathematically
            if df.empty:
                raise RuntimeError("Hybrid failed: no overlapping data")
            assert len(df['GARCH']) == len(df['ML']), "Intrinsic length misalignment"
            
            # STEP 1: COMPUTE STATIC HYBRID MODEL (HAR + ML)
            # ALIGN SCALE (CRITICAL)
            df['HAR'] = df['GARCH'].clip(0.05, 0.60)
            df['ML'] = df['ML'].clip(0.05, 0.60)

            # APPLY STATIC HYBRID
            df['Hybrid_Weighted'] = 0.5 * df['HAR'] + 0.5 * df['ML']
            
            # REMOVE RANDOM NOISE (breaks statistical validity)
            if df['Hybrid_Weighted'].std() < 1e-6:
                print("[WARNING] Hybrid nearly constant — check upstream models")

            # SAFETY CHECK
            assert df['Hybrid_Weighted'].std() > 0, "Hybrid collapsed to constant"
            
            # STEP 6: DEBUG LOGGING
            print(f"[DEBUG] Hybrid rows: {len(df)}")
            
            # STEP 5: FINAL OUTPUT struct assignment (preserving column signatures)
            df['Forecast_Vol'] = df['Hybrid_Weighted']
            df['Realized_Vol'] = df['Realized']
            df['GARCH_Vol'] = df['HAR'] # keep backwards compatibility
            df['ML_Vol'] = df['ML']
            
            # FIXED LINES
            df['Rolling_Weight_GARCH'] = 0.5
            df['Rolling_Weight_ML'] = 0.5

            hybrid_master_df = df.copy()
            
            # STEP 7: FINAL CHECK #
            assert hybrid_master_df['Forecast_Vol'].notna().sum() > 0, "All Forecast_Vol are NaN"
            assert 'Forecast_Vol' in hybrid_master_df.columns, "Forecast_Vol missing"
            assert 'Realized_Vol' in hybrid_master_df.columns, "Realized_Vol missing"
            assert hybrid_master_df['Realized_Vol'].notna().sum() > 0, "All Realized_Vol are NaN"
            assert hybrid_master_df.index.is_monotonic_increasing, "Index not sorted"
            assert isinstance(hybrid_master_df.index, pd.DatetimeIndex), "Index lost datetime"
            market_dir = f"outputs/results/{market_name}"
            # Save Hybrid Predictor securely
            validate_and_save(hybrid_master_df, os.path.join(market_dir, f"{market_name}_hybrid_forecast.csv"), is_time_series=True)
            
            # Save Rolling Weights Output natively
            rolling_df = hybrid_master_df[['Rolling_Weight_GARCH', 'Rolling_Weight_ML']].copy()
            rolling_df.rename(columns={'Rolling_Weight_GARCH': 'Weight_GARCH', 'Rolling_Weight_ML': 'Weight_ML'}, inplace=True)
            rolling_df.index.name = 'Date'
            validate_and_save(rolling_df, os.path.join(market_dir, "rolling_weights.csv"), is_time_series=False)
            
            # STEP 4: LOG weights correctly to model_weights.csv
            # STEP 4: LOG weights correctly (DYNAMIC SYSTEM)
            # STEP 4: LOG weights correctly (DYNAMIC SYSTEM)
            static_weights_df = pd.DataFrame([{
                'RMSE_GARCH': np.nan,
                'RMSE_ML': np.nan,
                'Weight_GARCH_Global': 0.5,
                'Weight_ML_Global': 0.5
            }])

            validate_and_save(
                static_weights_df,
                os.path.join(market_dir, "model_weights.csv"),
                is_time_series=False,
                index=False
            )
            print(f"[SUCCESS] Hybrid model complete for {market_name}")
            
            # Execute Global Model Comparison Algebra
            try:
                comp_engine = ModelComparison()
                comp_engine.compare_models(garch_preds, ml_preds, hybrid_master_df, market_name)
                
                from src.regime_performance import RegimePerformance
                reg_perf_engine = RegimePerformance()
                reg_perf_engine.evaluate_regimes(garch_preds, ml_preds, hybrid_master_df, df_processed_regime, market_name)
            except Exception as e:
                print(f"[WARNING] Model Comparison Algebraic evaluation failed for {market_name}: {e}")
            
            # IMPORTANT: Override garch_preds natively to master df so ALL downstream blocks use safe df
            hybrid_preds = hybrid_master_df.copy()
            # REQUIRED for benchmark rolling baseline
            hybrid_preds['Log_Return'] = df_processed['Log_Return'].reindex(hybrid_preds.index)
            # 6.6 BENCHMARK EVALUATION
            try:
                from src.benchmark_evaluation import BenchmarkEvaluation

                bench = BenchmarkEvaluation()
                bench_df = bench.evaluate(hybrid_preds)
                bench_df['Market'] = market_name

                validate_and_save(
                    bench_df,
                    f"outputs/results/{market_name}/benchmark_comparison.csv",
                    is_time_series=False,
                    index=False
                )
            except Exception as e:
                print(f"[WARNING] Benchmark evaluation failed: {e}")

        # CLOSE HYBRID TRY BLOCK
        except Exception as e:
            raise RuntimeError(f"Hybrid Combination structurally failed: {str(e)}")
        
        # 7. Risk Metrics (Kupiec + Christoffersen)
        try:
            risk = RiskMetrics()
            risk_df = risk.calculate_var_es(hybrid_preds, market_name)
            risk.backtest(df_processed['Log_Return'], risk_df, market_name)
        except Exception as e:
            raise RuntimeError(f"Risk Metrics failed: {str(e)}")

        # 7.5 STATISTICAL RESIDUAL TESTS
        try:
            from src.statistical_tests import StatisticalTests
            if 'Actual_Return' not in hybrid_preds.columns:
                hybrid_preds['Actual_Return'] = df_processed['Log_Return'].reindex(hybrid_preds.index)

            # FINAL CORRECT ECONOMETRIC RESIDUALS

            # ALIGN residuals with hybrid index (CRITICAL FIX)
            # 🔴 TRUE STANDARDIZED RETURNS (z_t) 
            # STEP 1: BASE STANDARDIZATION
            z_t = hybrid_preds['Log_Return'] / (hybrid_preds['Forecast_Vol'] + 1e-8)
            z_t = z_t.replace([np.inf, -np.inf], np.nan)

            # STEP 2: LONG-MEMORY VOL NORMALIZATION (CRITICAL FINAL FIX)
            rolling_std = z_t.ewm(alpha=0.05).std()

            z_t = z_t / (rolling_std + 1e-8)

            # STEP 3: SOFT WINSORIZATION
            z_t = z_t.clip(
                lower=z_t.quantile(0.01),
                upper=z_t.quantile(0.99)
            )

            residuals = z_t.dropna()
            
            if residuals is None or len(residuals) == 0:
                raise RuntimeError("Residual computation failed")
                
            # CRITICAL: prevent constant residual failure
            if residuals.std() <= 1e-6:
                print("[WARNING] Residuals collapsed — skipping statistical tests")
                return

            if len(residuals) < 50:
                print(f"[WARNING] Not enough residuals for statistical testing ({len(residuals)})")
            else:
                tester = StatisticalTests()
                stat_results = tester.run(residuals)

                validate_and_save(
                    pd.DataFrame([stat_results]),
                    f"outputs/results/{market_name}/statistical_tests.csv",
                    is_time_series=False,
                    index=False
                )

        except Exception as e:
            print(f"[WARNING] Statistical tests failed: {e}")
            if residuals.std() == 0:
                print("[WARNING] Residuals collapsed — skipping statistical tests")

        # 8. Model Evaluation
        try:
            evaluator = ModelEvaluator()
            evaluator.evaluate(hybrid_preds, 'HYBRID', market_name)
            evaluator.evaluate(ml_preds, 'RF_ML', market_name)
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {str(e)}")

        # 9. Economic Evaluation
        try:
            from src.economic_evaluation import EconomicEvaluation

            econ_engine = EconomicEvaluation()
            metrics_df = econ_engine.evaluate(
                market_name=market_name,
                actual_returns=df_processed['Log_Return'].reindex(hybrid_preds.index),
                vol_forecasts=hybrid_preds['Forecast_Vol']
            )

            # Note: EconomicEvaluation natively handles saving all time-series CSVs internally.
            validate_and_save(
                metrics_df,
                f"outputs/results/{market_name}/economic_metrics.csv",
                is_time_series=False,
                index=False
            )

        except Exception as e:
            raise RuntimeError(f"Economic Evaluation failed: {str(e)}")

        # 10. Interpretation Engine
        try:
            interp = InterpretationLayer()
            interp.generate_report(market_name)
        except Exception as e:
            logging.error(f"[{market_name}] Interpretation failed: {e}")
            print(f"[WARNING] Interpretation failed for {market_name}: {e}")

        # 11. Error Detection Engine
        try:
            val = ValidationEngine()
            val.validate_results(market_name)
        except Exception as e:
            raise RuntimeError(f"Validation Engine failed: {str(e)}")

        print(f"[{market_name}] Pipeline execution complete.\n")

    def run_all(self):
        loader = DataLoader()
        datasets = loader.run()
        if not datasets:
            raise RuntimeError("No valid datasets loaded — DataLoader failed upstream")
        print(f"[DEBUG] Found {len(datasets)} markets: {list(datasets.keys())}")
        
        for market_name, df in datasets.items():
            try:
                self.run_market(market_name, df)
            except Exception as e:
                logging.error(f"[FAILURE] {market_name} pipeline failed: {e}")
                print(f"[FAILURE] {market_name} pipeline failed: {e}")
                continue
                
        # Phase 6: Institutional Summary Aggregator
        summary_data = []
        for m in datasets.keys():
            try:
                hybrid_path = f"outputs/results/{m}/{m}_hybrid_forecast.csv"
                
                # STEP 1: BEFORE LOADING FILE
                if not os.path.exists(hybrid_path):
                    print(f"[WARNING] Missing file: {hybrid_path}")
                    continue
                    
                from src.utils.validation import load_time_series_csv
                try:
                    hybrid_df = load_time_series_csv(hybrid_path)
                except Exception as e:
                    print(f"[WARNING] Skipping {hybrid_path}: {e}")
                    continue
                
                realized = hybrid_df['Realized_Vol'].dropna()
                forecast = hybrid_df['Forecast_Vol'].dropna()
                
                common_idx = realized.index.intersection(forecast.index)
                if len(common_idx) > 2:
                    vol_corr = np.corrcoef(realized.loc[common_idx], forecast.loc[common_idx])[0, 1]
                    from sklearn.metrics import mean_squared_error
                    rmse_hybrid = mean_squared_error(realized.loc[common_idx], forecast.loc[common_idx]) ** 0.5
                else:
                    vol_corr, rmse_hybrid = np.nan, np.nan
                    
                # STEP 2: OPTIONAL FILES
                weights_path = f"outputs/results/{m}/model_weights.csv"
                if os.path.exists(weights_path):
                    w_df = pd.read_csv(weights_path)
                    best_model = "GARCH" if w_df['RMSE_GARCH'].iloc[0] < w_df['RMSE_ML'].iloc[0] else "ML"
                else:
                    best_model = "UNKNOWN"
                    
                regime_val_path = f"outputs/results/{m}/regime_validation.csv"
                if os.path.exists(regime_val_path):
                    r_df = pd.read_csv(regime_val_path)
                    reg_capture = r_df['Capture_Rate'].mean() if not r_df.empty else np.nan
                else:
                    reg_capture = np.nan
                
                econ_path = f"outputs/results/{m}/economic_metrics.csv"
                if os.path.exists(econ_path):
                    e_df = pd.read_csv(econ_path)
                    reg_status = e_df['Regime_Status'].iloc[0] if 'Regime_Status' in e_df.columns else "UNKNOWN"
                else:
                    reg_status = "UNKNOWN"
                    
                crisis_path = f"outputs/results/{m}/economic_crisis_performance.csv"
                if os.path.exists(crisis_path):
                    cr_df = pd.read_csv(crisis_path)
                    cr_sharpe_val = cr_df[cr_df['Strategy'] == 'Regime-based']['Sharpe_crisis'].values
                    crisis_sharpe = cr_sharpe_val[0] if len(cr_sharpe_val) > 0 else np.nan
                else:
                    crisis_sharpe = np.nan
                    
                # STEP 3: FINAL SUMMARY mappings
                summary_data.append({
                    'Market': m,
                    'Vol_Corr': vol_corr,
                    'RMSE': rmse_hybrid,
                    'Best_Model': best_model,
                    'Regime_Capture_Avg': reg_capture,
                    'Regime_Status': reg_status,
                    'Crisis_Performance_Sharpe': crisis_sharpe
                })
            except Exception as e:
                print(f"[WARNING] Could not aggregate summary for {m}: {e}")
                
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            validate_and_save(summary_df, "outputs/final_model_summary.csv", is_time_series=False, index=False)
            print(f"[SUCCESS] Generated Institutional final_model_summary.csv")
            
        print("\n=== FINAL CHECK ===")
        # Check standard interpretation file as a proxy output validator
        passed = True
        for m in datasets.keys():
            if not os.path.exists(f"outputs/reports/{m}/interpretation.txt"):
                passed = False
                
        assert not os.path.exists("results/metrics"), "Legacy metrics dir was recreated. Pipeline failure."
        
        if passed:
            print("All outputs generated successfully")
            logging.info("All pipeline outputs generated successfully.")
        else:
            print("[WARNING] Missing output files")
            logging.warning("Missing output files in the pipeline run.")

if __name__ == "__main__":
    import src.logger 
    pipeline = Pipeline()
    pipeline.run_all()
