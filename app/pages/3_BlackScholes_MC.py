import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from src.simulators import black_scholes_price, mc_option_price, bs_greeks

st.set_page_config(page_title="Black-Scholes MC", layout="wide")
st.title("💹 Black-Scholes — MC vs Analítico")
st.markdown(r"""
Precificamos uma opção europeia pelos dois métodos e comparamos o resultado.

$$C = S\,\Phi(d_1) - K e^{-rT}\Phi(d_2) \qquad d_1 = \frac{\ln(S/K)+(r+\sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$
""")
st.divider()

with st.sidebar:
    st.header("⚙️ Parâmetros")
    option_type = st.radio("Tipo", ["call", "put"], horizontal=True)
    S     = st.number_input("Spot S",              value=100.0, min_value=1.0)
    K     = st.number_input("Strike K",             value=105.0, min_value=1.0)
    T     = st.slider("Vencimento T (anos)",        0.1, 3.0, 1.0, 0.1)
    r     = st.slider("Taxa livre de risco r",      0.0, 0.15, 0.05, 0.005, format="%.3f")
    sigma = st.slider("Volatilidade σ (anual)",     0.05, 1.0, 0.20, 0.01, format="%.2f")
    n_sim = st.select_slider("Simulações MC",
        [10_000, 50_000, 100_000, 200_000, 500_000], value=100_000)
    seed  = st.number_input("Seed", value=42, min_value=0)

price_bs            = black_scholes_price(S, K, T, r, sigma, option_type)
price_mc, se        = mc_option_price(S, K, T, r, sigma, option_type, int(n_sim), int(seed))
greeks              = bs_greeks(S, K, T, r, sigma)
error_abs           = abs(price_mc - price_bs)
error_pct           = error_abs / price_bs * 100 if price_bs > 1e-6 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Black-Scholes analítico", f"${price_bs:.4f}")
c2.metric("Monte Carlo",             f"${price_mc:.4f}", f"SE ±{se:.4f}")
c3.metric("Erro absoluto",           f"${error_abs:.4f}")
c4.metric("Erro relativo",           f"{error_pct:.3f}%")
st.divider()

tab1, tab2, tab3 = st.tabs(["📉 Preço vs Strike", "🔢 Greeks", "ℹ️ Convergência MC"])

with tab1:
    strikes   = np.linspace(S * 0.5, S * 1.5, 120)
    bs_prices = [black_scholes_price(S, k, T, r, sigma, option_type) for k in strikes]
    intrinsic = (np.maximum(S - strikes, 0) if option_type == "call"
                 else np.maximum(strikes - S, 0))
    time_val  = np.array(bs_prices) - intrinsic

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=bs_prices, mode="lines",
        name="Preço BS", line=dict(color="#185FA5", width=2.5)))
    fig.add_trace(go.Scatter(x=strikes, y=intrinsic, mode="lines",
        name="Valor intrínseco", line=dict(color="#BA7517", dash="dash", width=1.5)))
    fig.add_trace(go.Scatter(x=strikes, y=time_val, mode="lines",
        name="Valor temporal", line=dict(color="#1D9E75", dash="dot", width=1.5)))
    fig.add_vline(x=K, line_dash="dot", line_color="#E24B4A",
        annotation_text=f"K = {K}", annotation_font_color="#E24B4A")
    fig.add_vline(x=S, line_dash="dot", line_color="white",
        annotation_text=f"S = {S}", annotation_position="top left")
    fig.update_layout(height=400, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="white", xaxis_title="Strike K", yaxis_title="Preço ($)",
        title=f"Decomposição do preço — {option_type.upper()}",
        legend=dict(bgcolor="#1a1d27"))
    fig.update_xaxes(showgrid=True, gridcolor="#2d3748")
    fig.update_yaxes(showgrid=True, gridcolor="#2d3748")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    g = greeks
    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
    gc1.metric("Delta Δ", f"{g['delta']:.4f}", help="∂C/∂S — hedge ratio")
    gc2.metric("Gamma Γ", f"{g['gamma']:.5f}", help="∂²C/∂S² — convexidade do delta")
    gc3.metric("Vega ν",  f"{g['vega']:.4f}",  help="∂C/∂σ por 1% de vol")
    gc4.metric("Theta Θ", f"{g['theta']:.4f}", help="∂C/∂t por dia (time decay)")
    gc5.metric("Rho ρ",   f"{g['rho']:.4f}",   help="∂C/∂r por 1% de taxa")
    st.divider()

    # Delta smile vs spot
    spots  = np.linspace(S * 0.5, S * 1.5, 120)
    deltas = [bs_greeks(s, K, T, r, sigma)["delta"] for s in spots]
    gammas = [bs_greeks(s, K, T, r, sigma)["gamma"] for s in spots]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=spots, y=deltas, mode="lines",
        name="Delta", line=dict(color="#1D9E75", width=2), yaxis="y1"))
    fig2.add_trace(go.Scatter(x=spots, y=gammas, mode="lines",
        name="Gamma", line=dict(color="#BA7517", width=2, dash="dash"), yaxis="y2"))
    fig2.add_vline(x=K, line_dash="dot", line_color="#E24B4A", annotation_text="ATM")
    fig2.update_layout(
        height=320, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="white", xaxis_title="Spot S",
        yaxis=dict(title="Delta", gridcolor="#2d3748"),
        yaxis2=dict(title="Gamma", overlaying="y", side="right", gridcolor="#2d3748"),
        legend=dict(bgcolor="#1a1d27"),
        title="Delta e Gamma em função do Spot",
    )
    fig2.update_xaxes(showgrid=True, gridcolor="#2d3748")
    st.plotly_chart(fig2, use_container_width=True)
    st.info("**Gamma máximo** ocorre ATM (At-The-Money, S≈K) — onde a opcionalidade é maior e o delta muda mais rapidamente.")

with tab3:
    ns_conv = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    mc_prices = []
    mc_ses    = []
    for n in ns_conv:
        p, s = mc_option_price(S, K, T, r, sigma, option_type, n, 42)
        mc_prices.append(p)
        mc_ses.append(s)
    mc_prices = np.array(mc_prices)
    mc_ses    = np.array(mc_ses)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=ns_conv, y=mc_prices + 2*mc_ses, mode="lines",
        line=dict(color="#185FA5", width=0), showlegend=False, fill=None))
    fig3.add_trace(go.Scatter(x=ns_conv, y=mc_prices - 2*mc_ses, mode="lines",
        line=dict(color="#185FA5", width=0), fill="tonexty",
        fillcolor="rgba(24,95,165,0.25)", name="IC 95%"))
    fig3.add_trace(go.Scatter(x=ns_conv, y=mc_prices, mode="lines+markers",
        name="Preço MC", line=dict(color="#1D9E75", width=2)))
    fig3.add_hline(y=price_bs, line_dash="dash", line_color="#BA7517",
        annotation_text=f"BS analítico = ${price_bs:.4f}")
    fig3.update_layout(height=350, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="white", xaxis_title="Nº de simulações",
        yaxis_title="Preço ($)", xaxis_type="log",
        title="Convergência do MC para o preço BS",
        legend=dict(bgcolor="#1a1d27"))
    fig3.update_xaxes(showgrid=True, gridcolor="#2d3748")
    fig3.update_yaxes(showgrid=True, gridcolor="#2d3748")
    st.plotly_chart(fig3, use_container_width=True)
    st.info(f"O IC 95% do MC encolhe como 1/√n. Com {n_sim:,} simulações, SE = ±{se:.4f} (≈{error_pct:.3f}% do preço).")