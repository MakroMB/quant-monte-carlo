import streamlit as st

st.set_page_config(page_title="Quant Monte Carlo Lab", page_icon="📊", layout="wide")

st.title("📊 Quant Monte Carlo Lab")
st.markdown("#### Simulações estocásticas aplicadas a finanças quantitativas")
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Módulos", "4", "ativos")
col2.metric("Simulações (máx)", "10⁶", "configurável")
col3.metric("BS vs analítico", "~0.03%", "erro MC")
col4.metric("Stack", "Python 3.11+")

st.divider()
st.markdown("""
### Navegue pelos módulos na barra lateral:

| Módulo | Tópico | Conceitos |
|---|---|---|
| 🎯 **Estimativa de π** | MC clássico | LLN, convergência, erro estocástico |
| 📈 **GBM Simulator** | Processos estocásticos | dS = μSdt + σSdW, trajetórias |
| 💹 **Black-Scholes MC** | Precificação de opções | Call/Put, Greeks, MC vs analítico |
| 🛡️ **VaR & CVaR** | Gestão de risco | Expected Shortfall, quantis |

---
> Projeto desenvolvido como portfólio para vagas **quant / risk / research**.  
> Stack: `Python` · `NumPy` · `SciPy` · `Streamlit` · `Plotly`
""")