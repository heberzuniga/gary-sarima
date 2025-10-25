# ===================== README.md =====================
- Evaluación de criterios AIC, MAPE y pruebas estadísticas (Jarque-Bera, Ljung-Box, ARCH).
- Dashboard visual de rendimiento.
- Generación automática de informes PDF institucionales.


---


## ⚙️ Instalación local


1. Clona este repositorio o descárgalo en tu entorno local.
2. Instala dependencias:
```bash
pip install -r requirements.txt
```
3. Ejecuta la aplicación:
```bash
streamlit run app.py
```


---


## ☁️ Despliegue en Streamlit Cloud


1. Crea un nuevo proyecto en [streamlit.io/cloud](https://streamlit.io/cloud).
2. Sube los siguientes archivos:
- `app.py`
- `requirements.txt`
3. Guarda y publica tu aplicación.


---


## 🧩 Funcionalidades principales


- **Menú lateral interactivo:** configuración de limpieza, Fourier, exógenas y periodos.
- **Grid Search inteligente:** selección automática de modelos ARIMA/SARIMA/SARIMAX.
- **Dashboard visual:** métricas clave, gráficos Top 10, AIC vs MAPE y pronóstico real vs predicho.
- **Informe PDF:** incluye título institucional, parámetros del experimento, gráficos y firma SolverTic SRL.


---


## 📄 Ejemplo de informe PDF
El reporte incluye:
- Fecha de generación
- Periodo analizado y parámetros del experimento
- Mejor modelo y desempeño (MAPE, AIC)
- Gráficos de validación y pronóstico
- Firma institucional de SolverTic SRL


---


## 👨‍💻 Autor
**Ing. Tito Zúñiga**
SolverTic SRL – División de Inteligencia Artificial y Modelado Predictivo
📧 contacto@solvertic.com


---


## 🏢 Derechos
© 2025 SolverTic SRL – Todos los derechos reservados.