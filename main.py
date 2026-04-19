import argparse
from src.pipeline import Pipeline
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Volatility Forecasting and Risk Modeling - China Markets Focus")
    parser.add_argument('--run-pipeline', action='store_true', help="Execute full data, econometrics, ML, and GARCH pipeline")
    parser.add_argument('--run-gui', action='store_true', help="Launch PyQt6 Research Dashboard")
    
    args = parser.parse_args()
    
    if args.run_pipeline:
        print("Starting Research Pipeline execution...")
        import src.pipeline
        import src.data_loader
        print(f"[DEBUG PATH] Pipeline: {src.pipeline.__file__}")
        print(f"[DEBUG PATH] DataLoader: {src.data_loader.__file__}")
        pipe = Pipeline()
        pipe.run_all()
        print("Pipeline Execution Complete. Artifacts saved in data/ and results/ directories.")
        
    elif args.run_gui:
        print("Launching GUI Dashboard...")
        # Since PyQt6 requires an isolated event loop, calling it directly or via subprocess is preferred.
        from src.gui import main as gui_main
        gui_main()
        
    else:
        print("No action specified. Use --run-pipeline or --run-gui.")
        parser.print_help()

if __name__ == "__main__":
    main()
