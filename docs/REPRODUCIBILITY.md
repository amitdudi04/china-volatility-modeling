# Reproducibility

## Environment and verification

The Python 3.11 research pipeline and its archived input data are maintained on the repository's default `main` branch.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py --verify-v22
```

The verification command checks raw-price source identities, complete per-market prediction files, numerical summary manifests, risk diagnostics and the integrity of saved research data. The current archived inputs are in `data/raw/`; the authoritative V2.2 results, source manifests and self-contained offline dashboard are in `outputs/research_v22/`. Technical hashes belong in those manifests, rather than the academic narrative.

## Computational reproduction

```bash
python main.py --run-pipeline
```

This command reruns the archived forecasting models, model-loss comparisons and the initial Gaussian-risk/strategy sensitivity into a **new** output directory, without overwriting the accepted research files. It does **not** rerun the separate Student-t VaR/ES extension. That extension uses `scripts/run_student_t_v22.py` for per-origin chunks and `scripts/summarize_student_t_v22.py` for checked aggregation; both must be executed with their documented input/output arguments and non-overwriting directories. Re-estimation is computationally expensive. The saved V2.2 release can be checked without refitting, but full empirical reproduction requires rerunning these separate components and reconciling all results.

```bash
python main.py --run-gui
```

This opens the offline browser-based dashboard without requiring external financial-data services. The separate Qt interface is optional and should not be treated as validated on every device.

The original raw CSVs and accepted results are preserved. The 2024 period is a historically examined, retrospective walk-forward evaluation—not an untouched holdout. For analytical definitions see [Methodology](METHODOLOGY.md), and for the reported tables see [Results](RESULTS.md).
