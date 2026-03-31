import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.simulators import estimate_pi, pi_convergence

st.set_page_config(page_title="Estimativa de π", layout="wide")
st.title("🎯 Estimativa de π — Monte Carlo")
st.markdown(r"""
**Ideia:** jogamos $n$ pontos aleatórios num quadrado $[0,1]^2$.  
A fração que cai dentro do quarto de círculo unitário converge para $\pi/4$.

$$\hat{\pi} = 4 \cdot \frac{\text{pontos dentro do círculo}}{n} \qquad \text{erro} \sim O\!\left(\frac{1}{\sqrt{n}}\right)$$
""")
st.divider()

with st.sidebar:
    st.header("⚙️ Parâmetros")
    n_samples = st.select_slider("Número de amostras",
        options=[1_000, 5_000, 10_000, 50_000, 100_000, 500_000], value=10_000)
    seed = st.number_input("Seed aleatória", value=42, min_value=0)

pi_est, x, y, inside = estimate_pi(n_samples, seed=int(seed))
ns, pi_vals = pi_convergence(max_samples=n_samples, seed=int(seed))
error_pct = abs(pi_est - np.pi) / np.pi * 100

c1, c2, c3 = st.columns(3)
c1.metric("π estimado", f"{pi_est:.6f}")
c2.metric("π real (numpy)", f"{np.pi:.6f}")
c3.metric("Erro relativo", f"{error_pct:.4f}%")
st.divider()

fig = make_subplots(rows=1, cols=2,
    subplot_titles=("Pontos amostrados", "Convergência de π"))

max_plot = min(n_samples, 5_000)
idx = np.random.choice(n_samples, max_plot, replace=False)
colors = np.where(inside[idx], "#1D9E75", "#E24B4A")

fig.add_trace(go.Scatter(x=x[idx], y=y[idx], mode="markers",
    marker=dict(color=colors, size=2, opacity=0.5), showlegend=False), row=1, col=1)

theta = np.linspace(0, np.pi / 2, 200)
fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines",
    line=dict(color="white", width=1.5), showlegend=False), row=1, col=1)

fig.add_trace(go.Scatter(x=ns, y=pi_vals, mode="lines",
    line=dict(color="#185FA5", width=2), name="π estimado"), row=1, col=2)
fig.add_hline(y=np.pi, line_dash="dash", line_color="#BA7517",
    annotation_text="π real", row=1, col=2)

fig.update_layout(height=450, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font_color="white", margin=dict(t=40, b=20))
fig.update_xaxes(showgrid=True, gridcolor="#2d3748", zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor="#2d3748", zeroline=False)
fig.update_xaxes(type="log", row=1, col=2, title_text="n amostras")
fig.update_yaxes(row=1, col=2, range=[2.8, 3.5])

st.plotly_chart(fig, use_container_width=True)
st.info(
    f"**Lei dos Grandes Números:** conforme n → ∞, o estimador converge para π. "
    f"O erro padrão cai como 1/√n — com {n_samples:,} amostras, erro ≈ **{error_pct:.4f}%**. "
    f"Para ganhar 1 dígito extra de precisão, você precisa de 100× mais amostras."
)