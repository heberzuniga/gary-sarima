# ==============================================================
# 🧠 Sistema Inteligente de Modelado del Precio de la Soya
# SolverTic SRL – División de Inteligencia Artificial y Modelado Predictivo
# Optimizado para Streamlit Cloud (sin multiprocessing)
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
from fpdf import FPDF
import warnings
warnings.filterwarnings("ignore")

# ========================= CONFIGURACIÓN STREAMLIT =========================
st.set_page_config(page_title="Sistema Inteligente de Modelado del Precio de la Soya", layout="wide")

# ========================= FUNCIONES AUXILIARES =========================
def limpiar_serie(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.interpolate(method="linear").bfill().ffill()
    return s

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

def fit_model(y, order, seasonal_order):
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res

def diagnosticos(res):
    resid = res.resid.dropna()
    jb_p = jarque_bera(resid)[1]
    lb_p = acorr_ljungbox(resid, lags=[min(24, len(resid)//2)], return_df=True)["lb_pvalue"].iloc[0]
    arch_p = het_arch(resid, nlags=12)[1]
    return jb_p, lb_p, arch_p

def buscar_modelos(train, test, pmax=3, qmax=3, periodo=12):
    st.info("🔍 Iniciando búsqueda secuencial de modelos (modo Streamlit Cloud optimizado)...")
    results = []
    total = (pmax + 1) * (qmax + 1)
    bar = st.progress(0)

    for i, (p, q) in enumerate([(p, q) for p in range(pmax + 1) for q in range(qmax + 1)]):
        bar.progress(int((i + 1) / total * 100))
        order = (p, 1, q)
        seasonal_order = (p, 1, q, periodo)
        try:
            res = fit_model(train, order, seasonal_order)
            fc = res.get_forecast(steps=len(test)).predicted_mean
            jb_p, lb_p, arch_p = diagnosticos(res)
            results.append({
                'order': order,
                'seasonal': seasonal_order,
                'aic': res.aic,
                'mape': mape(test, fc),
                'jb_p': jb_p,
                'lb_p': lb_p,
                'arch_p': arch_p,
                'res': res,
                'forecast': fc
            })
        except Exception as e:
            print(f"Error modelo {p,q}: {e}")
            continue

    df = pd.DataFrame(results)
    if df.empty:
        st.error("No se encontraron modelos válidos.")
        return None, None

    df['valid'] = (df['jb_p'] > 0.05) & (df['lb_p'] > 0.05) & (df['arch_p'] > 0.05)
    best = df.sort_values(['valid', 'mape', 'aic'], ascending=[False, True, True]).iloc[0]
    return df, best

# ========================= INTERFAZ PRINCIPAL =========================
st.title("🧠 Sistema Inteligente de Modelado del Precio de la Soya")
st.caption("SolverTic SRL – División de Inteligencia Artificial y Modelado Predictivo")

with st.sidebar:
    st.header("📂 Cargar y Configurar")
    file = st.file_uploader("Sube tu archivo CSV de precios mensuales", type=['csv'])
    pmax = st.slider("Máx p/q", 1, 5, 3)
    periodo_estacional = st.number_input("Periodo estacional (meses)", 3, 24, 12)
    test_size = st.slider("Meses para Test", 6, 36, 24)
    fecha_inicio = st.date_input("Inicio de análisis", datetime.date(2010, 1, 1))
    fecha_fin = st.date_input("Fin de análisis", datetime.date(2025, 5, 31))
    st.markdown("---")
    st.caption("Desarrollado por SolverTic SRL – Ingeniería de Sistemas Inteligentes © 2025")

if file:
    df = pd.read_csv(file)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index(df.columns[0]).sort_index()

    serie = limpiar_serie(df.iloc[:, 0])
    serie = serie.loc[(serie.index >= str(fecha_inicio)) & (serie.index <= str(fecha_fin))]

    train = serie[:-test_size]
    test = serie[-test_size:]

    st.subheader("📈 Vista previa de datos")
    st.line_chart(serie)
    st.write(f"**Observaciones:** {len(serie)} | Train={len(train)} | Test={len(test)}")

    df_res, best = buscar_modelos(train, test, pmax=pmax, qmax=pmax, periodo=periodo_estacional)

    if df_res is not None:
        st.success("✅ Modelado completado exitosamente")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mejor MAPE", f"{best['mape']:.2f}%")
        c2.metric("AIC", f"{best['aic']:.1f}")
        c3.metric("Modelos válidos", f"{df_res['valid'].sum()}/{len(df_res)}")

        st.subheader("🏆 Top 10 modelos por MAPE")
        st.dataframe(df_res.sort_values('mape').head(10)[['order', 'seasonal', 'mape', 'aic']])

        fig, ax = plt.subplots()
        ax.scatter(df_res['aic'], df_res['mape'], alpha=0.7, color='seagreen')
        ax.set_xlabel('AIC')
        ax.set_ylabel('MAPE (%)')
        ax.set_title('Relación AIC vs MAPE')
        st.pyplot(fig)

        res_best = best['res']
        fc = best['forecast']
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        train.plot(ax=ax2, label='Train')
        test.plot(ax=ax2, label='Test')
        fc.plot(ax=ax2, label='Pronóstico', color='red')
        ax2.legend()
        st.pyplot(fig2)

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
            pdf.cell(0, 8, f"p/q máximo: {pmax}", ln=True)
            pdf.cell(0, 8, f"Periodo estacional: {periodo_estacional}", ln=True)
            pdf.cell(0, 8, f"Mejor modelo: {best['order']} con MAPE={best['mape']:.2f}% y AIC={best['aic']:.1f}", ln=True)

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig2.savefig(tmp.name, dpi=150, bbox_inches="tight")
            pdf.image(tmp.name, w=170)
            tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            pdf.output(tmp_pdf.name)

            with open(tmp_pdf.name, 'rb') as f:
                st.download_button("💾 Descargar Informe PDF", f, file_name="Informe_Modelado_Soya.pdf", mime="application/pdf")
else:
    st.warning("Por favor, sube un archivo CSV con tu serie de precios mensuales.")
