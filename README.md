# litreview

Literature Review tools for Master's Thesis in Electrical Engineering at University of Costa Rica.

## Project Structure

- `litreview/`: Core Python package containing analysis and visualization logic.
- `config.yaml`: Central configuration file for models, labels, sources, and paths.
- `scripts/`: Entry point scripts for the pipeline.
- `analysis.slurm`: SLURM script for running the analysis on a cluster.

## Setup

This project uses `pyproject.toml` for dependency management. You can install it in editable mode:

```bash
pip install -e .
```

## Usage

### 1. Run Analysis
To process the raw data and run the zero-shot classification:

```bash
python scripts/run_analysis.py
```
*Note: This requires a GPU and may take significant time depending on the number of papers.*

### 2. Generate Plots
To generate visualizations and filter the classified data:

```bash
python scripts/run_plots.py
```

## Dependencies

- pandas
- numpy
- matplotlib
- transformers
- torch
- datasets
