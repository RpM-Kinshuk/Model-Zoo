# Alpha Clustering Lab

This repo now includes a local dashboard for experimenting with unsupervised clustering over per-layer empirical spectral density metrics, focused on the `alpha` exponent stored in `svd_results/metrics/*.h5`.

## What it does

- Loads all `.h5` alpha metric files and keeps the per-layer structure for each model.
- Handles different model depths by resampling each module trace onto a normalized depth grid.
- Supports mixed architectures through either:
  - the common 7-module LLM schema, or
  - canonical module roles across different raw module names.
- Lets you compare multiple representations:
  - summary statistics over depth,
  - full normalized alpha profiles,
  - dynamic time warping over normalized profiles.
- Lets you compare multiple clustering backends:
  - K-Means,
  - Agglomerative clustering,
  - DBSCAN.
- Reports several internal quality checks for each run:
  - silhouette,
  - Dunn index,
  - between/within distance ratio,
  - Calinski-Harabasz,
  - Davies-Bouldin,
  - assignment/noise share and cluster size balance.
- Can enrich the dashboard with lineage metadata from `sampled_metadata.csv` so the projection hover cards show:
  - inferred base model,
  - source model,
  - base-model relation.
- Includes a lineage-oriented view that:
  - evaluates clustering against known base-model families,
  - shows cluster-level base-model center summaries,
  - renders a metric-driven base-model constellation layout where predicted clusters are primary and base-model centers are overlaid from metadata.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit.

## Notes

- By default the dashboard treats `alpha <= 0` as missing because those values are not physically meaningful for the HT-SR interpretation.
- The `DTW over normalized profiles` mode is intentionally more expensive. Keep the sampled model count modest for responsive experiments.
- The main data source is the `.h5` alpha matrices because many top-level CSV files are empty or incomplete.
