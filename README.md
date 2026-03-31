# 📊 Quant Monte Carlo Lab

> Interactive Monte Carlo simulations for quantitative finance, built with Python + Streamlit.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Modules

| # | Module | Key concepts |
|---|--------|-------------|
| 1 | 🎯 **Pi Estimation** | Law of Large Numbers, MC convergence, O(1/√n) error |
| 2 | 📈 **GBM Simulator** | Geometric Brownian Motion, drift, volatility, log-normal distribution |
| 3 | 💰 **Black-Scholes MC** | Risk-neutral pricing, option payoffs, Greeks (Δ Γ Θ ν ρ) |
| 4 | ⚠️ **VaR & CVaR** | Value at Risk, Expected Shortfall, coherent risk measures |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/your-username/quant-monte-carlo.git
cd quant-monte-carlo

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app/Home.py
```

Open http://localhost:8501 in your browser.

---

## Project structure

```
quant-monte-carlo/
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Pi_Estimation.py
│       ├── 2_GBM_Simulator.py
│       ├── 3_Black_Scholes_MC.py
│       └── 4_VaR_CVaR.py
├── src/
│   ├── __init__.py
│   └── simulators.py
├── tests/
│   └── test_simulators.py
├── requirements.txt
└── README.md
```

---

## Stack

- **NumPy / SciPy** — vectorised simulation and statistics
- **Plotly** — interactive charts
- **Streamlit** — web dashboard