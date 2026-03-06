# litreview

Literature Review tools for Master's Thesis in Electrical Engineering at University of Costa Rica.

## Project Structure

- `litreview/`: Core Python package containing analysis and visualization logic.
- `config.yaml`: Central configuration file for models, labels, sources, and paths.
- `data/`:
    - `raw/`: Original CSV files from databases (Google Scholar, Web of Science, IEEE, Scopus).
    - `processed/`: Classified and filtered CSV results.
- `results/plots/`: Generated visualizations.
- `scripts/`: Entry point scripts for the pipeline.
- `analysis.slurm`: SLURM script for running the analysis on a cluster.

## Setup

This project uses `pyproject.toml` for dependency management. You can install it in editable mode:

```bash
pip install -e .
```

## Usage

### 1. Run Analysis
To process data and run the zero-shot classification:

**Using local CSVs:**
```bash
python scripts/run_analysis.py --config config.yaml
```

**Using Zotero Library:**
1. Configure your Zotero credentials in `config.yaml`.
2. Run:
```bash
python scripts/run_analysis.py --config config.yaml --fetch-zotero
```

*Note: Classification requires a GPU and may take significant time.*

### 2. Generate Plots
To generate visualizations and filter the classified data:

```bash
python scripts/run_plots.py --config config.yaml
```

## Dependencies

- pandas
- numpy
- matplotlib
- transformers
- torch
- datasets
- pyzotero
- PyYAML
