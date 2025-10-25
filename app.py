# ==============================================================
# 🧠 Sistema Inteligente de Modelado del Precio de la Soya – SolverTic SRL
# Versión 4.1 con formato profesional (MAPE y AIC: dos enteros, dos decimales)
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tempfile
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from scipy import stats
from fpdf import FPDF
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Sistema Inteligente de Modelado del Precio de la Soya", layout="wide")

# ==============================================================
# FUNCIONES AUXILIARES
# ==============================================================

def winsorize_series(s, low_q=0.01, high_q=0.99):
    lo, hi = s.quantile(low_q), s.quantile(high_q)
    return s.clip(lower=lo, upper=hi)

def limpiar_serie(s, winsor=True):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if winsor:
        s = winsorize_series(s)
    s = s.interpolate(method="linear").bfill().ffill()
    return s

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

def fit_model(y, order, seasonal_order, exog=None):
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order, exog=exog,
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)

def diagnosticos(res):
    resid = res.resid.dropna()
    jb_p = jarque_bera(resid)[1]
    lb_p = acorr_ljungbox(resid, lags=[min(24, len(resid)//2)], return_df=True)["lb_pvalue"].iloc[0]
    arch_p = het_arch(resid, nlags=12)[1]
    return jb_p, lb_p, arch_p, resid

def select_differencing(y):
    try:
        return 1 if adfuller(y.dropna())[1] > 0.05 else 0
    except:
        return 0

def fourier_terms(index, period=12, K=1):
    t = np.arange(len(index))
    X = {}
    for k in range(1, K + 1):
        X[f'sin_{k}'] = np.sin(2 * np.pi * k * t / period)
        X[f'cos_{k}'] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(X, index=index)

# ==============================================================
# BÚSQUEDA DE MODELOS
# ==============================================================

def buscar_modelos(train, test, pmax, qmax, Pmax, Qmax, periodo, include_fourier, K_min, K_max):
    st.info("🔍 Buscando el mejor modelo ARIMA / SARIMA / SARIMAX (modo Cloud Optimizado)...")
    results = []
    d = select_differencing(train)
    total = (pmax+1)*(qmax+1)*(Pmax+1)*(Qmax+1)
    bar = st.progress(0)
    for i, (p, q, P, Q) in enumerate([(p, q, P, Q) for p in range(pmax+1)
                                      for q in range(qmax+1)
                                      for P in range(Pmax+1)
                                      for Q in range(Qmax+1)]):
        bar.progress(int((i+1)/total*100))
        order = (p, d, q)
        seasonal_order = (P, 1, Q, periodo)
        try:
            if include_fourier:
                for K in range(K_min, K_max+1):
                    Xtrain = fourier_terms(train.index, periodo, K)
                    Xtest = fourier_terms(test.index, periodo, K)
                    res = fit_model(train, order, seasonal_order, exog=Xtrain)
                    fc = res.get_forecast(steps=len(test), exog=Xtest).predicted_mean
                    jb_p, lb_p, arch_p, resid = diagnosticos(res)
                    results.append({
                        'order': order,
                        'seasonal': seasonal_order,
                        'fourier_K': K,
                        'aic': res.aic,
                        'mape': mape(test, fc),
                        'jb_p': jb_p, 'lb_p': lb_p, 'arch_p': arch_p,
                        'valid': (jb_p > 0.05) & (lb_p > 0.05) & (arch_p > 0.05),
                        'res': res, 'forecast': fc, 'resid': resid
                    })
            else:
                res = fit_model(train, order, seasonal_order)
                fc = res.get_forecast(steps=len(test)).predicted_mean
                jb_p, lb_p, arch_p, resid = diagnosticos(res)
                results.append({
                    'order': order, 'seasonal': seasonal_order, 'fourier_K': None,
                    'aic': res.aic, 'mape': mape(test, fc),
                    'jb_p': jb_p, 'lb_p': lb_p, 'arch_p': arch_p,
                    'valid': (jb_p > 0.05) & (lb_p > 0.05) & (arch_p > 0.05),
                    'res': res, 'forecast': fc, 'resid': resid
                })
        except Exception:
            continue
    if not results:
        st.error("No se encontraron modelos válidos.")
        return None, None
    df = pd.DataFrame(results)
    best = df.sort_values(['valid', 'mape', 'aic'], ascending=[False, True, True]).iloc[0]
    return df, best

# ==============================================================
# INTERFAZ PRINCIPAL
# ==============================================================

st.title("🧠 Sistema Inteligente de Modelado del Precio de la Soya")
st.caption("SolverTic SRL – División de Inteligencia Artificial y Modelado Predictivo")

with st.sidebar:
    st.header("📂 Cargar y Configurar")
    file = st.file_uploader("Sube tu archivo CSV de precios mensuales", type=['csv'])
    pmax = st.slider("Máx p/q", 1, 5, 3)
    Pmax = st.slider("Máx P/Q (estacional)", 0, 3, 1)
    include_fourier = st.checkbox("Incluir Fourier (SARIMAX)", value=True)
    K_min, K_max = st.slider("Rango K Fourier", 1, 6, (1,3))
    periodo_estacional = st.number_input("Periodo estacional (meses)", 3, 24, 12)
    test_size = st.slider("Meses para Test", 6, 36, 24)
    fecha_inicio = st.date_input("Inicio de análisis", datetime.date(2010, 1, 1))
    fecha_fin = st.date_input("Fin de análisis", datetime.date(2025, 5, 31))
    winsor = st.checkbox("Capar outliers (winsorizar)", value=True)
    st.caption("Desarrollado por SolverTic SRL – Ingeniería de Sistemas Inteligentes © 2025")

if file:
    df = pd.read_csv(file)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index(df.columns[0]).sort_index()
    serie = limpiar_serie(df.iloc[:, 0], winsor=winsor)
    serie = serie.loc[(serie.index >= str(fecha_inicio)) & (serie.index <= str(fecha_fin))]
    train = serie[:-test_size]
    test = serie[-test_size:]

    st.subheader("📈 Vista previa de datos")
    st.line_chart(serie)
    st.write(f"**Observaciones:** {len(serie)} | Train={len(train)} | Test={len(test)}")

    df_res, best = buscar_modelos(train, test, pmax, qmax=pmax, Pmax=Pmax, Qmax=Pmax,
                                  periodo=periodo_estacional, include_fourier=include_fourier,
                                  K_min=K_min, K_max=K_max)

    if df_res is not None:
        st.success("✅ Modelado completado exitosamente")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mejor MAPE", f"{best['mape']:06.2f}%")
        c2.metric("AIC", f"{best['aic']:06.2f}")
        c3.metric("Modelos válidos", f"{df_res['valid'].sum()}/{len(df_res)}")

        st.subheader("🏆 Top 10 modelos por MAPE")
        tabla = df_res.sort_values('mape').head(10)[['order', 'seasonal', 'fourier_K', 'mape', 'aic']].copy()
        tabla['mape'] = tabla['mape'].map(lambda x: f"{x:06.2f}")
        tabla['aic'] = tabla['aic'].map(lambda x: f"{x:06.2f}")
        st.dataframe(tabla)

        fig, ax = plt.subplots()
        ax.scatter(df_res['aic'], df_res['mape'], alpha=0.7, color='seagreen')
        ax.set_xlabel('AIC')
        ax.set_ylabel('MAPE (%)')
        ax.set_title('Relación AIC vs MAPE')
        st.pyplot(fig)

        res_best = best['res']
        fc = best['forecast']
        resid_best = best['resid']

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        train.plot(ax=ax2, label='Train')
        test.plot(ax=ax2, label='Test')
        fc.plot(ax=ax2, label='Pronóstico', color='red')
        ax2.legend()
        st.pyplot(fig2)

        # ================= PDF =================
        if st.button("📄 Generar Informe PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Sistema Inteligente de Modelado del Precio de la Soya", ln=True, align='C')
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, "SolverTic SRL – División de Inteligencia Artificial y Modelado Predictivo", ln=True, align='C')
            pdf.ln(10)
            pdf.cell(0, 8, f"Fecha de generación: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.cell(0, 8, f"Periodo analizado: {fecha_inicio} a {fecha_fin}", ln=True)
            pdf.cell(0, 8, f"Meses Test: {test_size}", ln=True)
            pdf.cell(0, 8, f"p/q máximo: {pmax} | P/Q máximo: {Pmax}", ln=True)
            pdf.cell(0, 8, f"Periodo estacional: {periodo_estacional}", ln=True)
            pdf.cell(0, 8, f"Fourier incluido: {include_fourier} (K={K_min}-{K_max})", ln=True)
            pdf.cell(0, 8, f"Winsorización: {winsor}", ln=True)
            pdf.cell(0, 8, f"Mejor modelo: {best['order']} con MAPE={best['mape']:06.2f}% y AIC={best['aic']:06.2f}", ln=True)

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig2.savefig(tmp.name, dpi=150, bbox_inches="tight")
            pdf.image(tmp.name, w=170)

            tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            pdf.output(tmp_pdf.name)

            with open(tmp_pdf.name, 'rb') as f:
                st.download_button("💾 Descargar Informe PDF", f,
                                   file_name="Informe_Modelado_Soya.pdf",
                                   mime="application/pdf")

else:
    st.warning("Por favor, sube un archivo CSV con tu serie de precios mensuales.")
