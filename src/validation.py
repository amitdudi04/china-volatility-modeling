from src.utils.validation import validate_and_save
import pandas as pd
import os

class ValidationEngine:
    def __init__(self, base_dir="outputs/results"):
        self.base_dir = base_dir

    def validate_results(self, market):
        market_dir = os.path.join(self.base_dir, market)
        val_path = os.path.join(market_dir, "validation_report.csv")
        var_comp_path = os.path.join(market_dir, "var_comparison.csv")
        # DATA PRESENCE CHECK
        if not os.path.exists(var_comp_path):
            raise RuntimeError(f"Missing risk file: {var_comp_path}")
        flags = []
        if os.path.exists(var_comp_path):
            var_df = pd.read_csv(var_comp_path)
            for _, row in var_df.iterrows():
                model = row['Model']
                k_p = float(row.get('Kupiec_pvalue', 0))
                c_p = float(row.get('Christoffersen_pvalue', 0))
                dq_p = float(row.get('DQ_pvalue', 0))
                
                # SECTION 3 — DIAGNOSTICS RESTRUCTURING (3-TIER)
                if k_p < 0.01 or c_p < 0.01:
                    status = "FAIL"
                elif k_p < 0.05 or c_p < 0.05:
                    status = "WARNING"
                else:
                    status = "PASS"
                
                # SECTION 4 — INTERPRETATION LAYER
                interpretation = "Acceptable Risk Coverage"
                if status == "FAIL":
                    interpretation = "Underestimation of extreme tail risk (Critical)"
                elif status == "WARNING":
                    interpretation = "Borderline exceedance rate (Monitor)"

                flags.append({
                    "Model": model,
                    "Kupiec_p": k_p,
                    "Christoffersen_p": c_p,
                    "DQ_p": dq_p,
                    "Final_Status": status,
                    "Interpretation": interpretation
                })
        else:
            # Fallback structure mapping natively
            flags.append({
                "Model": "No Risk Data",
                "Kupiec_p": 0.0,
                "Christoffersen_p": 0.0,
                "DQ_p": 0.0,
                "Final_Status": "FAIL"
            })
            
        # Save output structured exactly on rules
        df_flags = pd.DataFrame(flags)
        validate_and_save(df_flags, val_path, is_time_series=False, index=False)
        print(f"Validation completed for {market}. Output -> {val_path}")
        return df_flags
