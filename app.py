# ==============================================================
st.caption("Desarrollado por SolverTic SRL © 2025")


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


# Gráfico AIC vs MAPE
fig, ax = plt.subplots()
ax.scatter(df_res['aic'], df_res['mape'], alpha=0.7, color='seagreen')
ax.set_xlabel('AIC')
ax.set_ylabel('MAPE (%)')
ax.set_title('Relación AIC vs MAPE')
st.pyplot(fig)


# Pronóstico
res_best = best['res']
fc = best['forecast']
fig2, ax2 = plt.subplots(figsize=(10, 4))
train.plot(ax=ax2, label='Train')
test.plot(ax=ax2, label='Test')
fc.plot(ax=ax2, label='Pronóstico', color='red')
ax2.legend()
st.pyplot(fig2)


# PDF
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