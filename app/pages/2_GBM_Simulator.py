import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from src.simulators import simulate_gbm, gbm_stats

st.set_page_config(page_title="GBM Simulator", layout="wide")
st.title("📈 GBM — Geometric Brownian Motion")
st.markdown(r"""
Modelo padrão para preços de ações. A equação diferencial estocástica e sua solução:

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t \implies S_t = S_0 \exp\!\left[\left(\mu - \tfrac{\sigma^2}{2}\right)t + \sigma W_t\right]$$
""")
st.divider()

with st.sidebar:
    st.header("⚙️ Parâmetros GBM")
    S0      = st.number_input("Preço inicial S₀", value=100.0, min_value=1.0)
    mu      = st.slider("Drift μ (retorno anual)", -0.20, 0.40, 0.08, 0.01, format="%.2f")
    sigma   = st.slider("Volatilidade σ (anual)", 0.05, 0.80, 0.20, 0.01, format="%.2f")
    T       = st.slider("Horizonte (anos)", 0.25, 5.0, 1.0, 0.25)
    n_paths = st.select_slider("Nº de trajetórias", [50, 100, 200, 500, 1000], value=200)
    seed    = st.number_input("Seed", value=42, min_value=0)

t, paths = simulate_gbm(S0=S0, mu=mu, sigma=sigma, T=T,
    n_paths=int(n_paths), seed=int(seed))
stats = gbm_stats(paths)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Preço médio final",  f"${stats['mean']:.2f}")
c2.metric("Mediana final",      f"${stats['median']:.2f}")
c3.metric("P5 (pessimista)",    f"${stats['p5']:.2f}")
c4.metric("P95 (otimista)",     f"${stats['p95']:.2f}")
st.divider()

tab1, tab2 = st.tabs(["📉 Trajetórias", "📊 Distribuição final"])

with tab1:
    fig = go.Figure()
    alpha = max(0.06, min(0.35, 15 / n_paths))
    for path in paths:
        fig.add_trace(go.Scatter(x=t, y=path, mode="lines",
            line=dict(color="#185FA5", width=0.7),
            opacity=alpha, showlegend=False))
    # Mean path
    fig.add_trace(go.Scatter(x=t, y=paths.mean(axis=0), mode="lines",
        name="Média MC", line=dict(color="#1D9E75", width=2.5)))
    # Analytical expected value
    fig.add_trace(go.Scatter(x=t, y=S0 * np.exp(mu * t), mode="lines",
        name="E[Sₜ] = S₀ e^{μt}", line=dict(color="#BA7517", width=2, dash="dash")))
    fig.update_layout(height=430, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="white", xaxis_title="Tempo (anos)", yaxis_title="Preço (R$)",
        legend=dict(bgcolor="#1a1d27"))
    fig.update_xaxes(showgrid=True, gridcolor="#2d3748")
    fig.update_yaxes(showgrid=True, gridcolor="#2d3748")
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        f"**Nota:** E[Sₜ] = S₀·e^(μt) mas a mediana = S₀·e^((μ-σ²/2)t). "
        f"A diferença é o **Jensen's inequality** — a distribuição log-normal é assimétrica."
    )

with tab2:
    final = paths[:, -1]
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=final, nbinsx=60,
        marker_color="#185FA5", opacity=0.8, name="Preços finais"))
    fig2.add_vline(x=stats["mean"],   line_dash="dash", line_color="#1D9E75",
        annotation_text=f"Média ${stats['mean']:.0f}")
    fig2.add_vline(x=stats["median"], line_dash="dash", line_color="#BA7517",
        annotation_text=f"Mediana ${stats['median']:.0f}")
    fig2.add_vline(x=stats["p5"],     line_dash="dot",  line_color="#E24B4A",
        annotation_text="P5")
    fig2.add_vline(x=stats["p95"],    line_dash="dot",  line_color="#9F77DD",
        annotation_text="P95")
    fig2.update_layout(height=430, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="white", xaxis_title="Preço final Sₜ", yaxis_title="Frequência")
    fig2.update_xaxes(showgrid=True, gridcolor="#2d3748")
    fig2.update_yaxes(showgrid=True, gridcolor="#2d3748")
    st.plotly_chart(fig2, use_container_width=True)
    st.info(
        f"Distribuição de Sₜ é **log-normal**. "
        f"Média ({stats['mean']:.2f}) > Mediana ({stats['median']:.2f}) — assimetria positiva. "
        f"Prob. de lucro: **{stats['prob_profit']*100:.1f}%**."
    )