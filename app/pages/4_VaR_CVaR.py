import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from src.simulators import compute_var_cvar

st.set_page_config(page_title="VaR & CVaR", layout="wide")
st.title("🛡️ VaR & CVaR — Gestão de Risco")
st.markdown(r"""
**Value at Risk:** com confiança $\alpha$, o portfólio não perde mais que VaR em $(1{-}\alpha)\%$ dos dias.  
**CVaR / Expected Shortfall:** a perda **média** nos piores $(1{-}\alpha)\%$ dos cenários.

$$\text{CVaR}_\alpha = \mathbb{E}\!\left[-R \mid R \leq -\text{VaR}_\alpha\right]$$
""")
st.divider()

with st.sidebar:
    st.header("⚙️ Parâmetros")
    portfolio_value = st.number_input("Valor do portfólio (R$)", value=1_000_000, step=100_000)
    mu_daily    = st.slider("Retorno médio diário μ", -0.005, 0.005, 0.0005, 0.0001, format="%.4f")
    sigma_daily = st.slider("Volatilidade diária σ",  0.001,  0.05,  0.012,  0.001,  format="%.3f")
    confidence  = st.slider("Nível de confiança",     0.90,   0.99,  0.95,   0.01,   format="%.2f")
    horizon     = st.slider("Horizonte (dias)",       1, 30, 1)
    n_sim       = st.select_slider("Simulações MC",
        [1_000, 5_000, 10_000, 50_000], value=10_000)
    seed        = st.number_input("Seed", value=42, min_value=0)

# Simulate simple univariate normal returns
rng = np.random.default_rng(int(seed))
daily_rets   = rng.normal(mu_daily, sigma_daily, (int(n_sim), horizon))
period_rets  = daily_rets.sum(axis=1)

var, cvar    = compute_var_cvar(period_rets, confidence)

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"VaR {confidence:.0%} ({horizon}d)",  f"{var*100:.2f}%",
          f"R$ {var*portfolio_value:,.0f}")
c2.metric(f"CVaR {confidence:.0%} ({horizon}d)", f"{cvar*100:.2f}%",
          f"R$ {cvar*portfolio_value:,.0f}")
c3.metric("Retorno médio", f"{period_rets.mean()*100:.3f}%")
c4.metric("Desvio padrão", f"{period_rets.std()*100:.3f}%")
st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Distribuição", "📉 VaR vs horizonte", "🔢 Backtesting visual"])

with tab1:
    tail_mask = period_rets <= -var
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=period_rets[~tail_mask]*100, nbinsx=70,
        marker_color="#185FA5", opacity=0.85, name="Retornos normais"))
    fig.add_trace(go.Histogram(x=period_rets[tail_mask]*100, nbinsx=25,
        marker_color="#E24B4A", opacity=0.9, name=f"Cauda {(1-confidence)*100:.0f}%"))
    fig.add_vline(x=-var*100,  line_dash="dash", line_color="#BA7517",
        annotation_text=f"VaR = {var*100:.2f}%", annotation_font_color="#BA7517")
    fig.add_vline(x=-cvar*100, line_dash="dot",  line_color="#E24B4A",
        annotation_text=f"CVaR = {cvar*100:.2f}%", annotation_font_color="#E24B4A")
    fig.update_layout(height=400, barmode="overlay",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
        xaxis_title="Retorno (%)", yaxis_title="Frequência",
        title=f"Distribuição de retornos — {horizon} dia(s)",
        legend=dict(bgcolor="#1a1d27"))
    fig.update_xaxes(showgrid=True, gridcolor="#2d3748")
    fig.update_yaxes(showgrid=True, gridcolor="#2d3748")
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        f"**{tail_mask.mean()*100:.1f}%** das simulações estão na cauda (esperado {(1-confidence)*100:.1f}%). "
        f"O CVaR (R$ {cvar*portfolio_value:,.0f}) é mais conservador que o VaR porque captura a **magnitude** das perdas extremas."
    )

with tab2:
    horizons = list(range(1, 31))
    vars_h, cvars_h = [], []
    for h in horizons:
        r_h = np.random.default_rng(42).normal(mu_daily, sigma_daily, (5_000, h)).sum(axis=1)
        v, c = compute_var_cvar(r_h, confidence)
        vars_h.append(v * 100); cvars_h.append(c * 100)

    var_sqrt = vars_h[0] * np.sqrt(np.array(horizons))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=horizons, y=vars_h, mode="lines+markers",
        name=f"VaR {confidence:.0%}", line=dict(color="#BA7517", width=2)))
    fig2.add_trace(go.Scatter(x=horizons, y=cvars_h, mode="lines+markers",
        name=f"CVaR {confidence:.0%}", line=dict(color="#E24B4A", width=2)))
    fig2.add_trace(go.Scatter(x=horizons, y=var_sqrt.tolist(), mode="lines",
        name="VaR × √T (teórico)", line=dict(color="#1D9E75", dash="dash", width=1.5)))
    fig2.update_layout(height=380, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="white", xaxis_title="Horizonte (dias)", yaxis_title="Perda (%)",
        title="Escalonamento do VaR com o horizonte",
        legend=dict(bgcolor="#1a1d27"))
    fig2.update_xaxes(showgrid=True, gridcolor="#2d3748")
    fig2.update_yaxes(showgrid=True, gridcolor="#2d3748")
    st.plotly_chart(fig2, use_container_width=True)
    st.info("Para retornos i.i.d., o VaR escala com **√T** (regra de raiz do tempo). A linha verde confirma isso empiricamente.")

with tab3:
    # Simulate a time series of 504 daily returns (2 years)
    ts_rets = np.random.default_rng(99).normal(mu_daily, sigma_daily, 504)
    # Rolling VaR: compute on expanding window starting at 60 days
    rolling_var = []
    for i in range(60, len(ts_rets)):
        v, _ = compute_var_cvar(ts_rets[:i], confidence)
        rolling_var.append(v)

    days = list(range(60, len(ts_rets)))
    breaches = [ts_rets[i] < -rolling_var[i-60] for i in range(60, len(ts_rets))]
    breach_days = [d for d, b in zip(days, breaches) if b]
    breach_rets = [ts_rets[d]*100 for d in breach_days]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=list(range(len(ts_rets))), y=ts_rets*100,
        mode="lines", name="Retorno diário",
        line=dict(color="#185FA5", width=1), opacity=0.7))
    fig3.add_trace(go.Scatter(x=days, y=[-v*100 for v in rolling_var],
        mode="lines", name=f"VaR {confidence:.0%} rolling",
        line=dict(color="#BA7517", width=1.5, dash="dash")))
    fig3.add_trace(go.Scatter(x=breach_days, y=breach_rets,
        mode="markers", name="Breaches",
        marker=dict(color="#E24B4A", size=6, symbol="x")))
    fig3.update_layout(height=380, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="white", xaxis_title="Dia", yaxis_title="Retorno (%)",
        title=f"Backtesting visual — {len(breach_days)} breaches em 504 dias ({len(breach_days)/504*100:.1f}%)",
        legend=dict(bgcolor="#1a1d27"))
    fig3.update_xaxes(showgrid=True, gridcolor="#2d3748")
    fig3.update_yaxes(showgrid=True, gridcolor="#2d3748")
    st.plotly_chart(fig3, use_container_width=True)
    expected_breaches = int(504 * (1 - confidence))
    st.info(
        f"Em 504 dias, esperamos **~{expected_breaches} breaches** ({(1-confidence)*100:.0f}%). "
        f"Observamos **{len(breach_days)}**. "
        f"Essa verificação é chamada de **backtesting de VaR** — exigido por Basileia III."
    )