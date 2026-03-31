"""
simulators.py — Core Monte Carlo simulation engines.
All functions are pure numpy — no side effects, easy to test.
"""

import numpy as np
from scipy.stats import norm


# ── 1. PI ESTIMATION ──────────────────────────────────────────────────────────

def estimate_pi(n_samples: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n_samples)
    y = rng.uniform(0, 1, n_samples)
    inside = (x**2 + y**2) <= 1.0
    return 4 * inside.mean(), x, y, inside


def pi_convergence(max_samples: int = 100_000, steps: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    ns = np.unique(np.logspace(1, np.log10(max_samples), steps).astype(int))
    x = rng.uniform(0, 1, max_samples)
    y = rng.uniform(0, 1, max_samples)
    inside = (x**2 + y**2) <= 1.0
    pi_vals = np.array([4 * inside[:n].mean() for n in ns])
    return ns, pi_vals


# ── 2. GBM ────────────────────────────────────────────────────────────────────

def simulate_gbm(S0=100.0, mu=0.08, sigma=0.20, T=1.0, n_steps=252, n_paths=1000, seed=42):
    """
    Exact log-normal GBM solution:
        S(t+dt) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.hstack([np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)])
    return t, S0 * np.exp(log_paths)


def gbm_stats(paths: np.ndarray) -> dict:
    final = paths[:, -1]
    return {
        "mean":        float(final.mean()),
        "median":      float(np.median(final)),
        "std":         float(final.std()),
        "p5":          float(np.percentile(final, 5)),
        "p95":         float(np.percentile(final, 95)),
        "prob_profit": float((final > paths[:, 0]).mean()),
    }


# ── 3. BLACK-SCHOLES ──────────────────────────────────────────────────────────

def black_scholes_price(S, K, T, r, sigma, option_type="call") -> float:
    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def mc_option_price(S, K, T, r, sigma, option_type="call", n_sims=200_000, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sims)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)
    disc = np.exp(-r * T) * payoffs
    return float(disc.mean()), float(disc.std() / np.sqrt(n_sims))


def bs_greeks(S, K, T, r, sigma) -> dict:
    if T <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    return {
        "delta": float(norm.cdf(d1)),
        "gamma": float(pdf_d1 / (S * sigma * np.sqrt(T))),
        "theta": float((-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365),
        "vega":  float(S * pdf_d1 * np.sqrt(T) / 100),
        "rho":   float(K * T * np.exp(-r * T) * norm.cdf(d2) / 100),
    }


# ── 4. VAR & CVAR ─────────────────────────────────────────────────────────────

def compute_var_cvar(returns: np.ndarray, confidence: float = 0.95):
    """VaR and CVaR. Both returned as positive loss magnitudes."""
    alpha = 1 - confidence
    var   = float(-np.percentile(returns, alpha * 100))
    tail  = returns[returns <= -var]
    cvar  = float(-tail.mean()) if len(tail) > 0 else var
    return var, cvar


def simulate_portfolio_returns(weights, mu_vec, cov_matrix, n_days=252, n_sims=5_000, seed=42):
    rng = np.random.default_rng(seed)
    asset_returns = rng.multivariate_normal(mu_vec / 252, cov_matrix / 252, size=(n_sims, n_days))
    return (asset_returns * weights).sum(axis=2).flatten()