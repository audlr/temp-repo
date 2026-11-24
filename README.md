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

The app supports four demo scenarios (scenarios.py):
- Circle area: exact integral is pi * 0.15^2.
- Joint default of two firms: Gaussian copula, PD 2%, correlation 0.3.
- Interference fringes: f = 1 + cos(100(x + y)).
- Starlight: narrow 2-D Gaussian peak at (0.40, 0.40), sigma 0.03.

Choosing a scenario sets its demo seed and sample size.

The MISER implementation recursively splits the region, estimates variance
in candidate subregions using pilot samples, and allocates the remaining
sample budget according to volume-weighted standard deviation.
