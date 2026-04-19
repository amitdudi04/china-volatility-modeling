import os, glob

fixes = [
    # validation.py
    ('validate_and_save(df_flags, val_path, index=False, is_time_series=True)', 'validate_and_save(df_flags, val_path, is_time_series=False, index=False)'),
    ('validate_and_save(df_flags, val_path, is_time_series=True)', 'validate_and_save(df_flags, val_path, is_time_series=False, index=False)'), 
    
    # risk_metrics.py
    ('validate_and_save(df, os.path.join(market_dir, "risk_forecasts.csv", is_time_series=True))', 'validate_and_save(df, os.path.join(market_dir, "risk_forecasts.csv"), is_time_series=True)'),
    ('validate_and_save(diag_df, os.path.join(market_dir, "volatility_comparison.csv", is_time_series=False))', 'validate_and_save(diag_df, os.path.join(market_dir, "volatility_comparison.csv"), is_time_series=False)'),
    ('validate_and_save(comp_df, os.path.join(market_dir, "var_comparison.csv", is_time_series=False), index=False)', 'validate_and_save(comp_df, os.path.join(market_dir, "var_comparison.csv"), is_time_series=False, index=False)'),
    ('validate_and_save(res_df, os.path.join(market_dir, "var_results.csv", is_time_series=False), index=False)', 'validate_and_save(res_df, os.path.join(market_dir, "var_results.csv"), is_time_series=False, index=False)'),
    
    # regime_performance.py
    ('validate_and_save(perf_df, os.path.join(market_dir, "regime_performance.csv", is_time_series=True), index=False)', 'validate_and_save(perf_df, os.path.join(market_dir, "regime_performance.csv"), is_time_series=False, index=False)'),
    
    # regime_model.py
    ('validate_and_save(reg_df, os.path.join(get_market_path(market_name, is_time_series=True), f"{market_name}_regimes.csv"))', 'validate_and_save(reg_df, os.path.join(get_market_path(market_name), f"{market_name}_regimes.csv"), is_time_series=True)'),
    ('validate_and_save(capture_df, save_path, index=False, is_time_series=True)', 'validate_and_save(capture_df, save_path, is_time_series=False, index=False)'),
    
    # preprocessor.py
    ('validate_and_save(df_processed, out_path, is_time_series=True)', 'validate_and_save(df_processed, out_path, is_time_series=True)'),
    
    # pipeline.py
    ('validate_and_save(hybrid_master_df, os.path.join(market_dir, f"{market_name}_hybrid_forecast.csv", is_time_series=True))', 'validate_and_save(hybrid_master_df, os.path.join(market_dir, f"{market_name}_hybrid_forecast.csv"), is_time_series=True)'),
    ('validate_and_save(rolling_df, os.path.join(market_dir, "rolling_weights.csv", is_time_series=False))', 'validate_and_save(rolling_df, os.path.join(market_dir, "rolling_weights.csv"), is_time_series=False)'),
    ('validate_and_save(static_weights_df, os.path.join(market_dir, "model_weights.csv", is_time_series=False), index=False)', 'validate_and_save(static_weights_df, os.path.join(market_dir, "model_weights.csv"), is_time_series=False, index=False)'),
    ('validate_and_save(summary_df, "outputs/final_model_summary.csv", index=False, is_time_series=True)', 'validate_and_save(summary_df, "outputs/final_model_summary.csv", is_time_series=False, index=False)'),
    
    # model_comparison.py
    ('validate_and_save(comp_df, os.path.join(market_dir, "model_comparison.csv", is_time_series=False), index=False)', 'validate_and_save(comp_df, os.path.join(market_dir, "model_comparison.csv"), is_time_series=False, index=False)'),
    
    # ml_models.py
    ('validate_and_save(fi, os.path.join(market_dir, "feature_stability.csv", is_time_series=False), index=False)', 'validate_and_save(fi, os.path.join(market_dir, "feature_stability.csv"), is_time_series=False, index=False)'),
    ('validate_and_save(pipeline_df, os.path.join(market_dir, f"{market_name}_ml_forecast.csv", is_time_series=True))', 'validate_and_save(pipeline_df, os.path.join(market_dir, f"{market_name}_ml_forecast.csv"), is_time_series=True)'),
    
    # garch_models.py
    ('validate_and_save(metrics_df, os.path.join(get_market_path(market_name, is_time_series=True), f"{market_name}_garch_comparison.csv"))', 'validate_and_save(metrics_df, os.path.join(get_market_path(market_name), f"{market_name}_garch_comparison.csv"), is_time_series=True)'),
    
    # forecast_engine.py
    ('validate_and_save(forecast_df, os.path.join(get_market_path(market_name, is_time_series=True), f"{market_name}_garch_forecast.csv"))', 'validate_and_save(forecast_df, os.path.join(get_market_path(market_name), f"{market_name}_garch_forecast.csv"), is_time_series=True)'),
    
    # evaluator.py
    ('validate_and_save(df_res, out_path, index=False, is_time_series=True)', 'validate_and_save(df_res, out_path, is_time_series=False, index=False)'),
    
    # eda_econometrics.py
    ('validate_and_save(res_df, os.path.join(get_market_path(market_name, is_time_series=False), f"{market_name}_econometric_tests.csv"))', 'validate_and_save(res_df, os.path.join(get_market_path(market_name), f"{market_name}_econometric_tests.csv"), is_time_series=False)'),
    
    # economic_evaluation.py
    ('validate_and_save(res_df, os.path.join(market_dir, "economic_performance.csv", is_time_series=True), index=False)', 'validate_and_save(res_df, os.path.join(market_dir, "economic_performance.csv"), is_time_series=False, index=False)'),
    ('validate_and_save(ts_df, os.path.join(market_dir, f"{market_name}_economic_timeseries.csv", is_time_series=True))', 'validate_and_save(ts_df, os.path.join(market_dir, f"{market_name}_economic_timeseries.csv"), is_time_series=True)'),
]

for filepath in glob.glob(r"g:\Volatility Forecasting, Regime Dynamics & Risk Modeling using GARCH\src\**\*.py", recursive=True):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    for bad, good in fixes:
        content = content.replace(bad, good)
        
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {filepath}")
