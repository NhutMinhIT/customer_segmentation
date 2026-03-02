# Customer Segmentation Demo

Simple Streamlit demo for clustering the Mall Customers dataset.

## Files

- `app.py` — Streamlit app (clustering demo)
- `data/Mall_Customers.csv` — sample dataset
- `requirements.txt` — dependencies

## Prerequisites

- Python 3.8+ (recommended)
- A virtual environment (optional but recommended)

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Start the Streamlit app:

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

## Notes

- The app expects the dataset at `data/Mall_Customers.csv` by default. Change the path in the sidebar if needed.
- `requirements.txt` already includes the main dependencies: Streamlit, pandas, numpy, scikit-learn, scipy, matplotlib.

## Next steps (optional)

- Pin dependency versions in `requirements.txt`.
- Add tests and CI workflow.
