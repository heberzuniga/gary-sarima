# ==============================================================
# 🧠 Sistema Inteligente de Modelado del Precio de la Soya
# SolverTic SRL – División de Inteligencia Artificial y Modelado Predictivo
# Autor: Ing. Tito Zúñiga
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import tempfile
import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from joblib import Parallel, delayed
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from scipy import stats
from fpdf import FPDF
import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# CONFIGURACIÓN STREAMLIT
# ==============================================================
st.set_page_config(page_title="Sistema Inteligente de Modelado del Precio de la Soya", layout="wide")

# ==============================================================
# FUNCIONES AUXILIARES
# ==============================================================
def to_month_end_index(idx) -> pd.DatetimeIndex:
    if isinstance(idx, pd.PeriodIndex):
        return idx.to_timestamp('M')
    idx = pd.to_datetime(idx, errors="coerce")
    return idx.to_period('M').to_timestamp('M')

def replace_nonfinite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def winsorize_series(s: pd.Series, low_q=0.01, high_q=0.99):
    lo, hi = s.quantile(low_q), s.quantile(high_q)
    return s.clip(lower=lo, upper=hi)

def interpolate_series(s: pd.Series, method="linear"):
    return s.interpolate(method=method, limit_direction="both")

def clean_pipeline(s: pd.Series, do_winsor=True, do_interp=True, do_ffill=True, do_bfill=True):
    s = replace_nonfinite(s)
    if do_winsor: s = winsorize_series(s)
    if do_interp: s = interpolate_series(s)
    if do_ffill: s = s.ffill()
    if do_bfill: s = s.bfill()
    return s

def ensure_monthly_series(s: pd.Series):
    s = s.dropna().sort_index()
    s = s.resample('M').mean().dropna()
    s.index = to_month_end_index(s.index)
    return s

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

def fit_sarimax(y, order, seasonal_order=(0,0,0,0), exog=None):
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order, exog=exog,
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)

def diagnostics(res):
    resid = res.resid.dropna()
    jb_p = jarque_bera(resid)[1]
    lb_p = acorr_ljungbox(resid, lags=[min(24, len(resid)//2)], return_df=True)["lb_pvalue"].iloc[0]
    arch_p = het_arch(resid, nlags=12)[1]
    return {"jb_p": jb_p, "lb_p": lb_p, "arch_p": arch_p, "resid": resid}

def record_result(kind, order, seasonal_order, test, fc, diag, aic):
    return {
        "model": kind,
        "order": order,
        "seasonal_order": seasonal_order,
        "aic": aic,
        "mape": mape(test, fc),
        "jb_p": diag["jb_p"],
        "lb_p": diag["lb_p"],
        "arch_p": diag["arch_p"],
        "forecast": fc
    }

def select_differencing(y):
    try: return 1 if adfuller(y.dropna())[1] > 0.05 else 0
    except: return 0

def fourier_terms(index, period=12, K=1):
    t = np.arange(len(index))
    X = {}
    for k in range(1, K + 1):
        X[f'sin_{k}'] = np.sin(2 * np.pi * k * t / period)
        X[f'cos_{k}'] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(X, index=index)

# ==============================================================
# GRID SEARCH INTELIGENTE Y DASHBOARD
# ==============================================================
def grid_search_models(train, test, seasonal_period=12, max_pq=3, include_fourier=True, K_fourier=(1, 2)):
    is_cloud = os.environ.get("STREAMLIT_RUNTIME") is not None
    exec_mode = "Cloud" if is_cloud else "Normal"

    d = select_differencing(train)
    results = []

    combos = [(p, d, q, D) for p in range(max_pq + 1) for q in range(max_pq + 1) for D in [0, 1]]
    total = len(combos)
    bar = st.progress(0)
    
    def evaluate_combo(p, d, q, D, idx):
        bar.progress(int((idx + 1) / total * 100))
        order = (p, d, q)
        seasonal_order = (p, D, q, seasonal_period)
        try:
            res = fit_sarimax(train, order, seasonal_order)
            fc = res.get_forecast(steps=len(test)).predicted_mean
            diag = diagnostics(res)
            return record_result("SARIMA", order, seasonal_order, test, fc, diag, res.aic)
        except:
            return None

    parallel_results = [evaluate_combo(p, d, q, D, i) for i, (p, d, q, D) in enumerate(combos)]
    results = [r for r in parallel_results if r]

    if not results:
        st.warning("No se encontraron modelos válidos.")
        return {"summary": pd.DataFrame(), "best": None}

    df = pd.DataFrame(results)
    df["passes_all"] = df[["jb_p", "lb_p", "arch_p"]].ge(0.05).all(axis=1)
    best = df.sort_values(["passes_all", "mape", "aic"], ascending=[False, True, True]).iloc[0].to_dict()

    # --- Dashboard visual ---
    st.markdown("## 📊 Resumen de Rendimiento del Modelado")
    total_models = len(df)
    passed_models = df["passes_all"].sum()
    pct_passed = passed_models / total_models * 100
    best_mape, best_aic = best['mape'], best['aic']

    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Mejor MAPE (%)", f"{best_mape:.2f}")
    c2.metric("📉 AIC", f"{best_aic:.1f}")
    c3.metric("📊 Modelos válidos", f"{pct_passed:.1f}%")

    top10 = df.sort_values("mape").head(10)
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.barh(top10["model"].astype(str), top10["mape"], color='seagreen')
    ax1.invert_yaxis()
    ax1.set_title("Top 10 Modelos con menor MAPE")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.pie([passed_models, total_models - passed_models], labels=['Válidos', 'No válidos'], autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
    ax2.set_title("Distribución de Calidad Estadística")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.scatter(df['aic'], df['mape'], color='slateblue', alpha=0.7)
    ax3.set_xlabel('AIC')
    ax3.set_ylabel('MAPE (%)')
    ax3.set_title('Relación AIC vs MAPE')
    st.pyplot(fig3)

    # --- Pronóstico ---
    best_order = best['order']
    best_seasonal = best['seasonal_order']
    res_best = fit_sarimax(train, best_order, best_seasonal)
    fc = res_best.get_forecast(steps=len(test)).predicted_mean

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    train.plot(ax=ax4, label='Train')
    test.plot(ax=ax4, label='Test')
    fc.plot(ax=ax4, label='Pronóstico')
    ax4.legend()
    ax4.set_title('Pronóstico sobre la muestra de evaluación')
    st.pyplot(fig4)

    # --- PDF ---
    if st.button("📥 Descargar Informe PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Sistema Inteligente de Modelado del Precio de la Soya", ln=True, align='C')
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, "SolverTic SRL – División de Inteligencia Artificial y Modelado Predictivo", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(0, 8, f"Fecha de generación: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 8, f"Modelos evaluados: {total_models}", ln=True)
        pdf.cell(0, 8, f"Porcentaje de modelos válidos: {pct_passed:.1f}%", ln=True)
        pdf.cell(0, 8, f"Mejor MAPE: {best_mape:.2f}%", ln=True)
        pdf.cell(0, 8, f"Mejor AIC: {best_aic:.1f}", ln=True)
        pdf.ln(10)

        for fig in [fig1, fig2, fig3, fig4]:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
            pdf.image(tmp.name, w=170)
            pdf.ln(5)

        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, 'rb') as f:
            st.download_button("💾 Descargar Informe PDF", f, file_name="Informe_Modelado_Soya.pdf", mime="application/pdf")

    return {"summary": df, "best": best, "forecast": fc}

# ==============================================================
# INTERFAZ PRINCIPAL STREAMLIT
# ==============================================================
st.title("Sistema Inteligente de Modelado del Precio de la Soya")
st.caption("Criterios: Normalidad, No autocorrelación, No heterocedasticidad y MAPE mínimo.")

with st.sidebar:
    st.header("📂 Datos")
    uploaded = st.file_uploader("Sube el CSV de precios limpios de la soya", type=['csv'])
    max_pq = st.slider("Máx p y q", 1, 5, 3)
    seasonal_period = st.number_input("Periodo estacional (meses)", 4, 24, 12)
    K_min, K_max = st.slider("Fourier K (SARIMAX)", 1, 6, (1, 2))
    st.markdown("---")
    st.caption("Desarrollado por SolverTic SRL – Ingeniería de Sistemas Inteligentes © 2025")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index(df.columns[0]).sort_index()

    target_col = df.columns[0]
    series = clean_pipeline(df[target_col])
    series = ensure_monthly_series(series)

    train = series[:-24]
    test = series[-24:]

    st.write(f"**Observaciones:** {len(series)} | Train={len(train)} | Test={len(test)}")
    out = grid_search_models(train, test, seasonal_period=int(seasonal_period), max_pq=int(max_pq), K_fourier=range(K_min, K_max + 1))

else:
    st.warning("Por favor, carga un archivo CSV para comenzar.")
