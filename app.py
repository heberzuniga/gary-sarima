# ==============================================================
st.pyplot(fig_r)


fig_h, ax_h = plt.subplots(figsize=(6,3))
ax_h.hist(resid_best, bins=20, color='skyblue', edgecolor='black', alpha=0.7, density=True)
xmin, xmax = ax_h.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mean_resid, std_resid)
ax_h.plot(x, p, 'r', linewidth=2)
ax_h.set_title("Histograma de residuales con curva normal")
st.pyplot(fig_h)


fig_qq = plt.figure(figsize=(5,3))
stats.probplot(resid_best, dist="norm", plot=plt)
plt.title("Q–Q Plot de los residuales (EViews Style)")
st.pyplot(fig_qq)


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig_acf = plt.figure(figsize=(5,3))
plot_acf(resid_best, lags=min(24,len(resid_best)//2), ax=plt.gca())
plt.title("ACF de los residuales")
st.pyplot(fig_acf)


fig_pacf = plt.figure(figsize=(5,3))
plot_pacf(resid_best, lags=min(24,len(resid_best)//2), ax=plt.gca(), method='ywm')
plt.title("PACF de los residuales")
st.pyplot(fig_pacf)


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
    pdf.cell(0, 8, f"Mejor modelo: {best['order']} con MAPE={best['mape']:.2f}% y AIC={best['aic']:.1f}", ln=True)

    pdf.ln(8)
    pdf.cell(0, 8, "📈 Diagnóstico de los Residuales del Mejor Modelo (Estilo EViews)", ln=True)
    for i, row in df_norm.iterrows():
        pdf.cell(0, 8, f"{row['Estadístico']}: {row['Valor']}", ln=True)

    # --- Guía rápida de interpretación ---
    pdf.ln(6)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "📘 Interpretación de los Diagnósticos (Guía Rápida)", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8,
        "• Jarque–Bera (JB): Normalidad de los residuales → p > 0.05 = OK\n"
        "• Ljung–Box (LB): Independencia temporal → p > 0.05 = OK\n"
        "• ARCH: Varianza constante → p > 0.05 = OK\n"
        "Si las tres pruebas superan 0.05, el modelo es estadísticamente sólido."
    )

    # --- Gráfico de pronóstico ---
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig2.savefig(tmp.name, dpi=150, bbox_inches="tight")
    pdf.image(tmp.name, w=170)

    # --- Generar PDF temporal ---
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_pdf.name)

    # --- Botón de descarga ---
    with open(tmp_pdf.name, 'rb') as f:
        st.download_button("💾 Descargar Informe PDF", f,
                           file_name="Informe_Modelado_Soya.pdf",
                           mime="application/pdf")
else:
    st.warning("Por favor, sube un archivo CSV con tu serie de precios mensuales.")
