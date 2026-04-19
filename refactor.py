import os, glob, re

target_dir = r"g:\Volatility Forecasting, Regime Dynamics & Risk Modeling using GARCH\src"
files = glob.glob(f"{target_dir}/**/*.py", recursive=True)

non_ts_files = ['model_comparison.csv', 'feature_stability.csv', 'regime_validation.csv', 'model_weights.csv', 'var_comparison.csv', 'var_results.csv', 'economic_crisis_performance.csv', 'econometric_tests.csv', 'volatility_comparison.csv', 'validation_report.csv', 'rolling_weights.csv']

def get_ts_flag(line):
    for f in non_ts_files:
        if f in line:
            return "is_time_series=False"
    return "is_time_series=True"

for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if '.to_csv(' not in content:
        continue
        
    print(f"Refactoring {filepath}...")
    
    if 'from src.utils.validation import validate_and_save' not in content:
        content = 'from src.utils.validation import validate_and_save\n' + content
        
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if '.to_csv(' in line:
            match = re.search(r'([A-Za-z0-9_]+)\.to_csv\((.*?)\)', line)
            if match:
                df_name = match.group(1)
                args = match.group(2)
                flag = get_ts_flag(line)
                
                # Using kwarg mapping physically
                new_line = line[:match.start()] + f"validate_and_save({df_name}, {args}, {flag})" + line[match.end():]
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))

print("Replacement successfully executed across pipeline natively.")
