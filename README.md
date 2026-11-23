# Monte Carlo vs MISER

## Run locally

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

The app supports two test functions:
- Circle indicator: exact integral is pi * 0.15^2.
- Gaussian: the narrow Gaussian from the original Python code.

The MISER implementation recursively splits the region, estimates variance
in candidate subregions using pilot samples, and allocates the remaining
sample budget according to volume-weighted standard deviation.
