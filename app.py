# Sistema Inteligente de Modelado del Precio de la Soya
# Autor: Tito Zúñiga
# Version final con Grid Search Inteligente, Dashboard Visual y Generación de PDF

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

# ==========================================
# CONFIGURACIÓN STREAMLIT
# ==========================================
st.set_page_config(page_title="Sistema Inteligente de Modelado del Precio de la Soya", layout="wide")

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def to_month_end_index(idx) -> pd.DatetimeIndex:
    if isinstance(idx, pd.PeriodIndex):
        return idx.to_timestamp('M')
    idx = pd.to_datetime(idx, errors="coerce")
    return idx.to_period('M').to_timestamp('M')

def replace_nonfinite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def winsorize_series(s: pd.Series, low_q=0.01, high_q=0.99):
    s_clean = s.copy()
    lo, hi = s_clean.quantile(low_q), s_clean.quantile(high_q)
    capped = (s_clean < lo) | (s_clean > hi)
    return s_clean.clip(lower=lo, upper=hi), capped

def interpolate_series(s: pd.Series, method="linear"):
    s2 = s.interpolate(method=method, limit_direction="both")
    return s2, s.isna() & (~s2.isna())

def apply_log(s: pd.Series, mode="none"):
    if mode == "none": return s
    if mode == "log": return np.log(s.where(s > 0, np.nan))
    if mode == "log1p": return np.log1p(s.where(s >= -0.999999, np.nan))
    return s

def clean_pipeline(s: pd.Series, do_winsor=True, q_low=0.01, q_high=0.99,
                   do_interp=True, interp_method="linear", do_ffill=True, do_bfill=True,
                   log_mode="none"):
    s0 = replace_nonfinite(s)
    if do_winsor:
        s1, _ = winsorize_series(s0, q_low, q_high)
    else:
        s1 = s0.copy()
    if do_interp:
        s2, _ = interpolate_series(s1, interp_method)
    else:
        s2 = s1.copy()
    if do_ffill: s2 = s2.ffill()
    if do_bfill: s2 = s2.bfill()
    s3 = apply_log(s2, log_mode)
    return s3

def ensure_monthly_series(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    try:
        s_m = s.resample("M").mean().dropna()
    except: s_m = s
    s_m.index = to_month_end_index(s_m.index)
    return s_m

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
    lb_lags = min(24, max(2, int(np.sqrt(len(resid)))))
    arch_lags = min(12, max(2, int(np.sqrt(len(resid)) // 2)))
    return {"jb_p": jarque_bera(resid)[1], "lb_p": acorr_ljungbox(resid, lags=[lb_lags], return_df=True)["lb_pvalue"].iloc[0], "arch_p": het_arch(resid, nlags=arch_lags)[1], "resid": resid}

def record_result(kind, order, seasonal_order, exog_desc, test, fc, diag, aic, extra=None):
    out = {"model": kind, "order": order, "seasonal_order": seasonal_order, "exog": exog_desc, "aic": aic,
           "mape": mape(test, fc), "jb_p": diag["jb_p"], "lb_p": diag["lb_p"], "arch_p": diag["arch_p"], "forecast": fc}
    if extra: out.update(extra)
    return out

def select_differencing(y):
    y = pd.Series(y).dropna()
    try: return 1 if adfuller(y, autolag="AIC")[1] > 0.05 else 0
    except: return 0

def fourier_terms(index, period=12, K=1):
    t = np.arange(len(index))
    return pd.DataFrame({f"sin_{k}": np.sin(2*np.pi*k*t/period), f"cos_{k}": np.cos(2*np.pi*k*t/period)} for k in range(1,K+1)).sum()

# ==========================================
# GRID SEARCH INTELIGENTE
# ==========================================

def grid_search_models(train, test, seasonal_period=12, max_pq=3, exog_train=None, exog_test=None,
                        include_fourier=True, K_fourier=(1,2), mape_threshold=None, enforce_threshold=False, n_jobs=-1):
    is_cloud = os.environ.get("STREAMLIT_RUNTIME", None) is not None
    n_obs = len(train) + len(test)
    exec_mode = "Normal"
    if n_obs > 150 or max_pq > 3:
        exec_mode = "Rápido"
        max_pq = min(max_pq, 2)
        K_fourier = [1]
        st.info("⚡ **Modo rápido inteligente activado**")
    elif is_cloud:
        exec_mode = "Cloud"
        st.info("🌐 **Modo seguro activado (sin paralelización)**")
    else:
        st.info("🚀 **Modo normal activado (paralelizado)**")

    d = select_differencing(train)
    results = []
    fit_cache = {}
    def fit_cached(order, seasonal_order, use_exog, k=None):
        key = (order, seasonal_order, use_exog, k)
        if key in fit_cache: return fit_cache[key]
        try:
            Xtr = exog_train if use_exog and exog_train is not None else None
            Xte = exog_test if use_exog and exog_test is not None else None
            if k is not None:
                ft_tr = fourier_terms(train.index, period=seasonal_period, K=k)
                ft_te = fourier_terms(test.index, period=seasonal_period, K=k)
                Xtr = ft_tr if Xtr is None else pd.concat([Xtr, ft_tr], axis=1)
                Xte = ft_te if Xte is None else pd.concat([Xte, ft_te], axis=1)
            res = fit_sarimax(train, order=order, seasonal_order=seasonal_order, exog=Xtr)
            fc = res.get_forecast(steps=len(test), exog=Xte).predicted_mean
            diag = diagnostics(res)
            r = record_result("SARIMAX" if use_exog else "SARIMA", order, seasonal_order, f"exog={use_exog},K={k}", test, fc, diag, res.aic)
            fit_cache[key] = r; return r
        except: return None

    combos = [(p,d,q,D) for p in range(max_pq+1) for q in range(max_pq+1) for D in [0,1]]
    total = len(combos)
    bar = st.progress(0); info = st.empty()

    def evaluate(p,d,q,D,idx):
        rset = []
        for k in ([None]+list(K_fourier) if include_fourier else [None]):
            for ex in [False, True] if exog_train is not None else [False]:
                r = fit_cached((p,d,q),(p,D,q,seasonal_period),ex,k)
                if r: rset.append(r)
        bar.progress(int((idx+1)/total*100))
        info.text(f"Evaluando combinación {idx+1}/{total}")
        return rset

    import time
    t0=time.time()
    if is_cloud:
        par=[evaluate(p,d,q,D,i) for i,(p,d,q,D) in enumerate(combos)]
    else:
        try:
            par=Parallel(n_jobs=n_jobs)(delayed(evaluate)(p,d,q,D,i) for i,(p,d,q,D) in enumerate(combos))
        except:
            par=[evaluate(p,d,q,D,i) for i,(p,d,q,D) in enumerate(combos)]

    bar.progress(100); info.text(f"Búsqueda completada en {(time.time()-t0)/60:.1f} min")
    for rr in par: results.extend([r for r in rr if r])
    if not results: return {"summary":pd.DataFrame(),"best":None}

    df=pd.DataFrame(results)
    df["passes_all"]=df[["jb_p","lb_p","arch_p"]].ge(0.05).all(axis=1)
    df["passes_count"]=df[["jb_p","lb_p","arch_p"]].ge(0.05).sum(axis=1)
    df["meets_thresh"]=df["mape"]<= (mape_threshold if mape_threshold else np.inf)
    best=df.sort_values(["passes_all","passes_count","mape","aic"],ascending=[False,False,True,True]).iloc[0].to_dict()

    # === DASHBOARD VISUAL ===
    st.markdown("## 📊 Resumen de Rendimiento del Modelado")
    total_models=len(df); passed=df["passes_all"].sum(); pct_passed=passed/total_models*100
    best_mape=best["mape"]; best_aic=best["aic"]
    c1,c2,c3=st.columns(3)
    c1.metric("🏆 Mejor MAPE (%)",f"{best_mape:.2f}"); c2.metric("📉 AIC",f"{best_aic:.1f}"); c3.metric("📊 % válidos",f"{pct_passed:.1f}%")

    top10=df.sort_values("mape").head(10)
    fig,ax=plt.subplots(figsize=(8,4)); ax.barh(top10["model"],top10["mape"],color="seagreen"); ax.invert_yaxis(); ax.set_xlabel("MAPE (%)"); ax.set_title("Top 10 Modelos por MAPE"); st.pyplot(fig)
    fig2,ax2=plt.subplots(figsize=(4,4)); ax2.pie([passed,total_models-passed],labels=['Válidos','No válidos'],autopct='%1.1f%%',startangle=90,colors=['#66b3ff','#ff9999']); ax2.axis('equal'); st.pyplot(fig2)
    fig3,ax3=plt.subplots(figsize=(6,4)); ax3.scatter(df["aic"],df["mape"],color='slateblue',alpha=0.7); ax3.set_xlabel("AIC"); ax3.set_ylabel("MAPE (%)"); ax3.set_title("Relación AIC vs MAPE"); st.pyplot(fig3)

    # === PDF ===
    if st.button("📥 Descargar Informe PDF"):
        pdf=FPDF(); pdf.add_page(); pdf.set_font("Arial","B",16)
        pdf.cell(0,10,"Sistema Inteligente de Modelado del Precio de la Soya",ln=True,align='C')
        pdf.set_font("Arial","",12)
        pdf.cell(0,10,f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",ln=True)
        pdf.cell(0,10,f"Modelos evaluados: {total_models}",ln=True)
        pdf.cell(0,10,f"% Modelos válidos: {pct_passed:.1f}%",ln=True)
        pdf.cell(0,10,f"Mejor MAPE: {best_mape:.2f}% | Mejor AIC: {best_aic:.1f}",ln=True)
        pdf.ln(10)
        for f in [fig,fig2,fig3]:
            tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".png"); f.savefig(tmp.name,dpi=150,bbox_inches="tight"); pdf.image(tmp.name,w=170); pdf.ln(5)
        tmpfile=tempfile.NamedTemporaryFile(delete=False,suffix=".pdf"); pdf.output(tmpfile.name)
        with open(tmpfile.name,"rb") as f: st.download_button("💾 Descargar PDF",data=f,file_name="Informe_Modelado_Soya.pdf",mime="application/pdf")

    return {"summary":df,"best":best}

# ==========================================
# INTERFAZ PRINCIPAL STREAMLIT
# ==========================================

st.title("Sistema Inteligente de Modelado del Precio de la Soya")
st.caption("Criterios: Normalidad, No autocorrelación, No heterocedasticidad y MAPE mínimo.")

uploaded=st.file_uploader("📂 Sube el CSV de precios limpios de la soya",type=['csv'])
if uploaded is None:
    st.stop()
df=pd.read_csv(uploaded)

date_col=df.columns[0]; df[date_col]=pd.to_datetime(df[date_col]); df=df.set_index(date_col).sort_index()
num_cols=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not num_cols: st.error("No hay columnas numéricas."); st.stop()

series=df[num_cols[0]].astype(float)
series=ensure_monthly_series(series)
train=series[:-24]; test=series[-24:]

st.write(f"**Observaciones:** {len(series)} | Train={len(train)} | Test={len(test)}")

out=grid_search_models(train,test)

if out['best'] is not None:
    st.success(f"Mejor modelo: {out['best']['model']} | MAPE={out['best']['mape']:.2f}% | AIC={out['best']['aic']:.1f}")
