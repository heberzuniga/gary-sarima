# ===================== README.md (actualización) =====================


## 📘 Guía de Interpretación de los Diagnósticos Estadísticos


En el proceso de validación de modelos ARIMA, SARIMA o SARIMAX, se aplican tres pruebas estadísticas fundamentales para verificar la calidad de los residuales del modelo. Un modelo óptimo debe presentar **residuales que se comporten como ruido blanco**, es decir, sin correlación, con varianza constante y distribuidos normalmente.


### 1. Prueba Jarque–Bera (JB)
**Objetivo:** Verificar la **normalidad de los residuales**.
**Hipótesis nula (H₀):** Los residuales siguen una distribución normal.
**Criterio:** Si *p-value > 0.05*, no se rechaza H₀ → los residuales son normales.
**Interpretación:** Valores altos (p < 0.05) indican que los residuales no son normales, lo que puede afectar la validez estadística de los intervalos de confianza y las predicciones.


### 2. Prueba Ljung–Box (LB)
**Objetivo:** Detectar **autocorrelación en los residuales**.
**Hipótesis nula (H₀):** No existe autocorrelación (los errores son independientes).
**Criterio:** Si *p-value > 0.05*, no se rechaza H₀ → los residuales son independientes.
**Interpretación:** Si *p < 0.05*, los residuales presentan autocorrelación → el modelo no ha captado toda la estructura temporal y puede mejorarse con otros parámetros o términos estacionales.


### 3. Prueba ARCH (Heterocedasticidad condicional)
**Objetivo:** Evaluar si la **varianza de los residuales es constante** (homocedasticidad).
**Hipótesis nula (H₀):** No hay heterocedasticidad (la varianza es constante).
**Criterio:** Si *p-value > 0.05*, no se rechaza H₀ → los residuales son homocedásticos.
**Interpretación:** Si *p < 0.05*, los residuales presentan varianza condicional (heterocedasticidad), lo cual sugiere que puede ser útil emplear modelos ARCH/GARCH para capturar mejor la volatilidad.


---


### 📊 Interpretación global
Un modelo con **JB > 0.05**, **LB > 0.05** y **ARCH > 0.05** se considera **estadísticamente sólido**, ya que sus residuales se comportan como ruido blanco. Si alguna de las pruebas falla (p < 0.05), el modelo puede mejorarse ajustando parámetros o aplicando transformaciones.


---


### 📘 Interpretación de los Diagnósticos (Guía Rápida – incluida en PDF)
- **Jarque–Bera (JB):** Normalidad de los residuales → *p > 0.05 = OK*
- **Ljung–Box (LB):** Independencia temporal → *p > 0.05 = OK*
- **ARCH:** Varianza constante → *p > 0.05 = OK*


✅ Si las tres pruebas son mayores a 0.05, el modelo es estadísticamente adecuado.
⚠️ Si alguna es menor a 0.05, revisar especificaciones o considerar Fourier, diferenciación adicional o modelos con volatilidad (ARCH/GARCH).

