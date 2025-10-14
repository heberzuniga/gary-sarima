import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Soya ARIMA/SARIMA/SARIMAX Selector", layout="wide")

# ============================
# Utilidades de índice
# ============================

def to_month_end_index(idx) -> pd.DatetimeIndex:
    """Convierte cualquier índice a fin de mes (DatetimeIndex)."""
    if isinstance(idx, pd.PeriodIndex):
        return idx.to_timestamp('M')
    # DatetimeIndex u otra cosa convertible
    idx = pd.to_datetime(idx, errors="coerce")
    return idx.to_period('M').to_timestamp('M')

# ============================
# Limpieza de datos
# ============================

def replace_nonfinite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def winsorize_series(s: pd.Series, low_q=0.01, high_q=0.99):
    s_clean = s.copy()
    lo = s_clean.quantile(low_q)
    hi = s_clean.quantile(high_q)
    capped = (s_clean < lo) | (s_clean > hi)
    return s_clean.clip(lower=lo, upper=hi), capped

def interpolate_series(s: pd.Series, method="linear"):
    before = s.isna()
    s2 = s.interpolate(method=method, limit_direction="both")
    after = s2.isna()
    filled = before & (~after)
    return s2, filled

def apply_log(s: pd.Series, mode="none"):
    if mode == "none":
        return s
    if mode == "log":
        s_pos = s.where(s > 0, np.nan)
        return np.log(s_pos)
    if mode == "log1p":
        s_shift = s.where(s >= -0.999999, np.nan)
        return np.log1p(s_shift)
    return s

def clean_pipeline(s: pd.Series, do_winsor=True, q_low=0.01, q_high=0.99,
                   do_interp=True, interp_method="linear",
                   do_ffill=True, do_bfill=True,
                   log_mode="none"):
    report = {}
    s0 = replace_nonfinite(s)
    report["initial_na"] = int(s0.isna().sum())

    if do_winsor:
        s1, capped_mask = winsorize_series(s0, q_low, q_high)
    else:
        s1, capped_mask = s0.copy(), pd.Series(False, index=s0.index)
    report["winsor_applied"] = bool(do_winsor)

    if do_interp:
        s2, interp_mask = interpolate_series(s1, interp_method)
    else:
        s2, interp_mask = s1.copy(), pd.Series(False, index=s1.index)

    if do_ffill:
        pre_ffill_na = s2.isna()
        s2 = s2.ffill()
        ffill_mask = pre_ffill_na & (~s2.isna())
    else:
        ffill_mask = pd.Series(False, index=s2.index)

    if do_bfill:
        pre_bfill_na = s2.isna()
        s2 = s2.bfill()
        bfill_mask = pre_bfill_na & (~s2.isna())
    else:
        bfill_mask = pd.Series(False, index=s2.index)

    report["after_fill_na"] = int(s2.isna().sum())

    s3 = apply_log(s2, log_mode)
    report["transform"] = log_mode
    report["final_na"] = int(s3.isna().sum())
    report["length"] = int(len(s3))

    masks = {"capped": capped_mask, "interp": interp_mask, "ffill": ffill_mask, "bfill": bfill_mask}
    return s3, report, masks

# ============================
# Helpers de modelado y frecuencia
# ============================

def try_parse_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)

def find_date_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in ["fecha", "date", "period", "time", "month", "fecha_mes"]]
    for c in df.columns:
        if c not in candidates:
            parsed = try_parse_dates(df[c])
            if parsed.notna().mean() > 0.8:
                candidates.append(c)
    return candidates[0] if len(candidates) else None

def ensure_monthly_series(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    try:
        s_m = s.resample("M").mean().dropna()
    except Exception:
        s_m = s
    # fuerza fin de mes
    s_m.index = to_month_end_index(s_m.index)
    return s_m

def ensure_monthly_df(X: pd.DataFrame) -> pd.DataFrame:
    X = X.sort_index()
    try:
        X_m = X.resample("M").mean().dropna(how="all")
    except Exception:
        X_m = X
    X_m.index = to_month_end_index(X_m.index)
    return X_m

def train_test_split_monthly(s: pd.Series, eval_start="2023-01-01", eval_end="2025-05-31") -> Tuple[pd.Series, pd.Series]:
    s = ensure_monthly_series(s)
    train = s[s.index < pd.to_datetime(eval_start)]
    test  = s[(s.index >= pd.to_datetime(eval_start)) & (s.index <= pd.to_datetime(eval_end))]
    # normaliza índices a fin de mes (por si acaso)
    train.index = to_month_end_index(train.index)
    test.index  = to_month_end_index(test.index)
    return train, test

def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera

def fit_sarimax(y, order, seasonal_order=(0,0,0,0), exog=None):
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order, exog=exog,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res

def ljungbox_pvalue(residuals, lags=24):
    lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    return float(lb["lb_pvalue"].iloc[0])

def arch_pvalue(residuals, lags=12):
    stat, p, _, _ = het_arch(residuals, nlags=lags)
    return float(p)

def jb_pvalue(residuals):
    jb_stat, jb_p, _, _ = jarque_bera(residuals)
    return float(jb_p)

def fourier_terms(index, period=12, K=1):
    t = np.arange(len(index))
    X = {}
    for k in range(1, K+1):
        X[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        X[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(X, index=index)

def select_differencing(y: pd.Series) -> int:
    y_clean = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(y_clean) < 12:
        return 0
    if y_clean.std() == 0 or y_clean.max() == y_clean.min():
        return 0
    try:
        adf_p = adfuller(y_clean.values, autolag="AIC")[1]
        return 1 if adf_p > 0.05 else 0
    except Exception:
        return 0

def diagnostics(res) -> Dict[str, float]:
    resid = res.resid.dropna()
    lb_lags = min(24, max(2, int(np.sqrt(len(resid)))))
    arch_lags = min(12, max(2, int(np.sqrt(len(resid)) // 2)))
    return {"jb_p": jb_pvalue(resid),
            "lb_p": ljungbox_pvalue(resid, lags=lb_lags),
            "arch_p": arch_pvalue(resid, lags=arch_lags),
            "resid": resid}

def record_result(kind, order, seasonal_order, exog_desc, test, fc, diag, aic, extra: Optional[dict]=None):
    out = {
        "model": kind,
        "order": order,
        "seasonal_order": seasonal_order,
        "exog": exog_desc,
        "aic": aic,
        "mape": mape(test.values, fc.values) if (test is not None and fc is not None) else np.inf,
        "jb_p": diag["jb_p"], "lb_p": diag["lb_p"], "arch_p": diag["arch_p"],
        "forecast": fc
    }
    if extra: out.update(extra)
    return out

# ============================
# Auto-selección de exógenas (columnas + lags hasta 12)
# ============================

def auto_select_exog_and_lags(
    y_train: pd.Series,
    df_indexed: pd.DataFrame,
    candidate_cols: List[str],
    max_base_cols: int = 3,
    max_lag: int = 12,
    max_total_features: int = 12,
    min_abs_corr: float = 0.20,
    winsor=True, q_low=0.01, q_high=0.99,
    interp=True, interp_method="linear",
    ffill=True, bfill=True,
    standardize=False
):
    if not candidate_cols:
        return None, {}
    candidates = []  # (score, col, lag, series_shifted)
    for col in candidate_cols:
        s = df_indexed[col].astype(float)
        s_clean, _, _ = clean_pipeline(s, winsor, q_low, q_high, interp, interp_method, ffill, bfill, log_mode="none")
        s_m = ensure_monthly_series(s_clean)
        for L in range(0, max_lag + 1):
            x = s_m.shift(L).reindex(y_train.index)
            if x.isna().all() or x.std(ddof=0) == 0: continue
            r = x.corr(y_train)
            score = float(abs(r)) if pd.notna(r) else 0.0
            candidates.append((score, col, L, x))
    candidates.sort(key=lambda t: t[0], reverse=True)

    selected_series, selected_names = [], []
    chosen_map: dict[str, list[int]] = {}
    base_cols_used: set[str] = set()

    for score, col, L, x in candidates:
        if len(selected_series) >= max_total_features: break
        if col not in base_cols_used and len(base_cols_used) >= max_base_cols: continue
        if score < min_abs_corr and len(selected_series) > 0: continue
        if selected_series:
            corrs = [abs(x.corr(s)) for s in selected_series if s.std(ddof=0) > 0]
            if len(corrs) and max(corrs) > 0.95: continue
        selected_series.append(x)
        selected_names.append(f"{col}_lag{L}")
        chosen_map.setdefault(col, []).append(L)
        base_cols_used.add(col)

    if not selected_series:
        return None, {}
    X = pd.concat([s.rename(n) for s, n in zip(selected_series, selected_names)], axis=1)
    if standardize:
        X = (X - X.mean()) / X.std(ddof=0)
    X = ensure_monthly_df(X).reindex(y_train.index)
    return X, chosen_map

# ============================
# Grid search
# ============================
from joblib import Parallel, delayed
from functools import lru_cache

def grid_search_models(
    train: pd.Series,
    test: pd.Series,
    seasonal_period=12,
    max_pq=3,
    exog_train: Optional[pd.DataFrame] = None,
    exog_test: Optional[pd.DataFrame] = None,
    include_fourier=True,
    K_fourier=(1, 2),
    mape_threshold=None,
    enforce_threshold=False,
    n_jobs=-1
) -> Dict[str, Any]:
    """Optimized grid search for ARIMA/SARIMA/SARIMAX models (parallelized, early-stopping)."""
    
    d = select_differencing(train)
    results = []

    @lru_cache(maxsize=None)
    def fit_cached(order, seasonal_order, use_exog: bool, k: Optional[int]):
        try:
            Xtr = exog_train if use_exog and exog_train is not None else None
            Xte = exog_test if use_exog and exog_test is not None else None
            if k is not None:
                ft_tr = fourier_terms(train.index, period=seasonal_period, K=k)
                ft_te = fourier_terms(test.index,  period=seasonal_period, K=k)
                Xtr = ft_tr if Xtr is None else pd.concat([Xtr, ft_tr], axis=1)
                Xte = ft_te if Xte is None else pd.concat([Xte, ft_te], axis=1)
            res = fit_sarimax(train, order=order, seasonal_order=seasonal_order, exog=Xtr)
            fc = res.get_forecast(steps=len(test), exog=Xte).predicted_mean
            fc.index = test.index
            diag = diagnostics(res)
            return record_result(
                "SARIMAX" if use_exog else "SARIMA",
                order, seasonal_order,
                f"exog={use_exog}, K={k}",
                test, fc, diag, res.aic,
                extra={"fourier_k": k, "exog_cols": list(exog_train.columns) if use_exog and exog_train is not None else []}
            )
        except Exception:
            return None

    # --- Generación de combinaciones posibles ---
    combos = []
    for p in range(0, max_pq + 1):
        for q in range(0, max_pq + 1):
            for D in [0, 1]:
                combos.append((p, d, q, D))

    # --- Función auxiliar (con early-stop lógico) ---
    def evaluate_combo(p, d, q, D):
        combo_results = []
        seasonal_order = (p, D, q, seasonal_period)
        order = (p, d, q)

        # Sin exógenas ni Fourier
        r1 = fit_cached(order, seasonal_order, False, None)
        if r1: combo_results.append(r1)

        # Con exógenas si existen
        if exog_train is not None and exog_train.shape[1] > 0:
            r2 = fit_cached(order, seasonal_order, True, None)
            if r2: combo_results.append(r2)

        # Con Fourier o Fourier+Exógenas
        if include_fourier:
            for k in K_fourier:
                r3 = fit_cached(order, seasonal_order, False, k)
                if r3: combo_results.append(r3)
                if exog_train is not None and exog_train.shape[1] > 0:
                    r4 = fit_cached(order, seasonal_order, True, k)
                    if r4: combo_results.append(r4)

        # --- Early stopping: si todos los MAPE > 30% se omite rama ---
        if all([r["mape"] > 30 for r in combo_results if r is not None]):
            return []
        return combo_results

    # --- Ejecución paralela ---
    parallel_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(evaluate_combo)(p, d, q, D) for (p, d, q, D) in combos
    )

    # --- Consolidar resultados ---
    for rset in parallel_results:
        results.extend([r for r in rset if r is not None])

    if len(results) == 0:
        return {"summary": pd.DataFrame(), "best": None}

    df = pd.DataFrame(results)
    df["passes_all"] = df[["jb_p", "lb_p", "arch_p"]].ge(0.05).all(axis=1)
    df["passes_count"] = df[["jb_p", "lb_p", "arch_p"]].ge(0.05).sum(axis=1)
    df["meets_thresh"] = df["mape"] <= (mape_threshold if mape_threshold is not None else np.inf)

    # --- Selección del mejor modelo ---
    if mape_threshold is not None:
        valid = df[df["passes_all"] & df["meets_thresh"]]
        if len(valid) > 0:
            best_row = valid.sort_values(["mape", "aic"]).iloc[0].to_dict()
        else:
            best_row = df.sort_values(["passes_all", "passes_count", "mape", "aic"],
                                      ascending=[False, False, True, True]).iloc[0].to_dict()
            if df["meets_thresh"].any() and not best_row["meets_thresh"]:
                best_row["_note"] = "No hubo modelos que cumplan el MAPE objetivo; se muestra el mejor compromiso."
    else:
        best_row = df.sort_values(["passes_all", "passes_count", "mape", "aic"],
                                  ascending=[False, False, True, True]).iloc[0].to_dict()

    return {
        "summary": df.sort_values(["passes_all", "passes_count", "mape", "aic"],
                                  ascending=[False, False, True, True]),
        "best": best_row
    }


# ============================
# Plots & refits
# ============================

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_series(train, test, fc, title):
    fig, ax = plt.subplots(figsize=(10,4))
    train.plot(ax=ax, label="Train"); test.plot(ax=ax, label="Test (holdout)")
    if fc is not None: fc.plot(ax=ax, label="Forecast")
    ax.set_title(title); ax.legend(); st.pyplot(fig)

def plot_residuals(resid, title_prefix=""):
    fig, ax = plt.subplots(figsize=(10,3))
    resid.plot(ax=ax); ax.set_title(f"{title_prefix} Residuals over time"); st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(6,3)); ax.hist(resid, bins=20, alpha=0.7); ax.set_title(f"{title_prefix} Residuals Histogram"); st.pyplot(fig)
    fig = plt.figure(figsize=(6,3)); stats.probplot(resid, dist="norm", plot=plt); plt.title(f"{title_prefix} Q-Q plot"); st.pyplot(fig)
    fig_acf = plt.figure(figsize=(6,3)); plot_acf(resid, lags=min(24, len(resid)//2), ax=plt.gca()); plt.title(f"{title_prefix} Residuals ACF"); st.pyplot(fig_acf)
    fig_pacf = plt.figure(figsize=(6,3)); plot_pacf(resid, lags=min(24, len(resid)//2), ax=plt.gca(), method="ywm"); plt.title(f"{title_prefix} Residuals PACF"); st.pyplot(fig_pacf)

def fit_best_model_for_summary(best: dict, train: pd.Series, seasonal_period: int,
                               exog_train: Optional[pd.DataFrame]=None):
    order = tuple(best.get("order", (0,0,0)))
    seasonal_order = tuple(best.get("seasonal_order", (0,0,0,0)))
    X = exog_train if best.get("exog_cols") else None
    k = best.get("fourier_k", None)
    if k is not None:
        ft_train = fourier_terms(train.index, period=int(seasonal_period), K=int(k))
        X = ft_train if X is None else pd.concat([X, ft_train], axis=1)
    res = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                  exog=X, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return res

def refit_and_get_resid(best: dict, train: pd.Series, seasonal_period: int,
                        exog_train: Optional[pd.DataFrame]=None):
    res = fit_best_model_for_summary(best, train, seasonal_period, exog_train)
    return res.resid.dropna()

def forecast_best(best: dict, train: pd.Series, test: pd.Series, seasonal_period: int,
                  exog_train: Optional[pd.DataFrame], exog_test: Optional[pd.DataFrame]):
    order = tuple(best["order"]); seasonal_order = tuple(best["seasonal_order"])
    Xtr = exog_train if best.get("exog_cols") else None
    Xte = exog_test  if best.get("exog_cols") else None
    k = best.get("fourier_k", None)
    if k is not None:
        ft_train = fourier_terms(train.index, period=int(seasonal_period), K=int(k))
        ft_test  = fourier_terms(test.index,  period=int(seasonal_period), K=int(k))
        Xtr = ft_train if Xtr is None else pd.concat([Xtr, ft_train], axis=1)
        Xte = ft_test  if Xte is None else pd.concat([Xte, ft_test], axis=1)
    res_tmp = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                      exog=Xtr, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = res_tmp.get_forecast(steps=len(test), exog=Xte).predicted_mean
    fc.index = test.index
    return fc

# ============================
# UI principal
# ============================

st.title("Modelado del Grano de Soya: ARIMA vs SARIMA vs SARIMAX")
st.caption("Criterios: normalidad, no autocorrelación, no heterocedasticidad y MAPE mínimo en la evaluación (2023-01 a 2025-05).")

with st.sidebar:
    st.header("Datos")
    default_path = "/mnt/data/soya_limpio_ddmmyyyy.csv"
    uploaded = st.file_uploader("Sube el CSV", type=["csv"])
    path = st.text_input("o ruta del CSV", value=default_path)
    eval_start = st.text_input("Inicio evaluación (YYYY-MM-DD)", value="2023-01-01")
    eval_end   = st.text_input("Fin evaluación (YYYY-MM-DD)", value="2025-05-31")
    max_pq = st.slider("Máx p y q", 1, 5, 3)
    seasonal_period = st.number_input("Periodo estacional (m)", min_value=4, max_value=24, value=12)
    K_min, K_max = st.slider("Fourier K (SARIMAX)", 1, 6, (1,3))

# Carga de datos
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    if not Path(path).exists():
        st.error(f"No se encontró el archivo: {path}"); st.stop()
    df = pd.read_csv(path)

st.subheader("Vista previa de datos")
st.dataframe(df.head(10))

date_col = find_date_col(df)
if date_col is None:
    st.error("No se pudo detectar la columna de fecha. Usa una columna como 'fecha' o 'date'."); st.stop()

df[date_col] = try_parse_dates(df[date_col])
df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

# Columnas numéricas disponibles
num_cols_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not num_cols_all:
    st.error("No hay columnas numéricas en el dataset."); st.stop()

# Objetivo (primera numérica por defecto)
target = num_cols_all[0]
st.write(f"**Fecha:** `{date_col}` | **Objetivo:** `{target}`")

# ---- Controles de limpieza (objetivo) ----
with st.sidebar:
    st.header("Limpieza de objetivo")
    do_winsor = st.checkbox("Capar outliers (winsorizar)", value=True)
    q_low, q_high = st.slider("Percentiles de capping", 0.0, 0.2, (0.01, 0.99))
    do_interp = st.checkbox("Interpolar huecos (lineal)", value=True)
    interp_method = st.selectbox("Método de interpolación", ["linear","time","nearest","polynomial"], index=0)
    do_ffill = st.checkbox("Relleno hacia adelante (ffill)", value=True)
    do_bfill = st.checkbox("Relleno hacia atrás (bfill)", value=True)
    log_mode = st.selectbox("Transformación", ["none","log","log1p"], index=0)

    st.header("Exógenas")
    exog_mode = st.selectbox("Modo de exógenas", ["Automático", "Manual", "Ninguna"], index=0)
    exog_standardize = st.checkbox("Estandarizar exógenas (z-score)", value=False)
    max_base_cols = st.number_input("Máx. columnas base (auto)", min_value=1, max_value=10, value=3)
    max_total_features = st.number_input("Máx. features finales (auto)", min_value=1, max_value=30, value=12)
    min_abs_corr = st.slider("Umbral |corr| mínimo (auto)", 0.0, 0.8, 0.20, 0.01)
    exog_cols_user = []
    if exog_mode == "Manual":
        exog_candidates = [c for c in num_cols_all if c != target]
        exog_cols_user = st.multiselect("Selecciona columnas exógenas (manual)", exog_candidates, default=[])
    include_fourier = st.checkbox("Añadir Fourier a SARIMAX", value=True,
                                  help="Si no se usan exógenas, SARIMAX probará Fourier solo. Si hay exógenas, probará exógenas+Fourier.")

    st.header("Faltantes en exógenas (test)")
    exog_missing_policy = st.selectbox(
        "Política ante faltantes/extrapolación",
        ["Permitir extrapolación (interpolar/ffill/bfill)",
         "No extrapolar: degradar a sin exógenas",
         "No extrapolar: fallar si faltan"], index=0
    )

    st.header("Robustez")
    robust_mode = st.checkbox("Modo robusto (auto-sanar target/exógenas)", value=True)
    min_train_months = st.number_input("Mínimo meses en Train", min_value=6, max_value=48, value=12)

    st.header("Criterio MAPE")
    mape_thr = st.number_input("Umbral MAPE objetivo (%)", min_value=0.1, max_value=50.0, value=4.0, step=0.1)
    enforce_thr = st.checkbox("Exigir umbral MAPE", value=True)

# ---- Limpieza + mensualización del objetivo ----
series_raw = df[target].astype(float)
series, clean_report, _ = clean_pipeline(series_raw, do_winsor, q_low, q_high, do_interp, interp_method, do_ffill, do_bfill, log_mode)

with st.expander("Reporte de limpieza (objetivo)", expanded=False):
    st.write({"NA iniciales": clean_report["initial_na"],
              "Winsor aplicado": clean_report["winsor_applied"],
              "NA después de interpolar/ffill/bfill": clean_report["after_fill_na"],
              "Transformación": clean_report["transform"],
              "NA finales": clean_report["final_na"],
              "Observaciones totales": clean_report["length"]})
    fig, ax = plt.subplots(figsize=(10,3))
    ensure_monthly_series(series_raw).plot(ax=ax, label="Crudo (mensualizado)")
    ensure_monthly_series(series).plot(ax=ax, label="Limpio (mensualizado)")
    ax.set_title("Validación de limpieza"); ax.legend(); st.pyplot(fig)

series = ensure_monthly_series(series).dropna()
# normaliza serie a fin de mes (por si acaso)
series.index = to_month_end_index(series.index)

train, test = train_test_split_monthly(series, eval_start=eval_start, eval_end=eval_end)

# ---- Exógenas (auto o manual; lags 0..12 automáticos) ----
X_all = None
exog_lags_map = {}
if exog_mode == "Automático":
    candidates = [c for c in num_cols_all if c != target]
    X_all, exog_lags_map = auto_select_exog_and_lags(
        y_train=train, df_indexed=df, candidate_cols=candidates,
        max_base_cols=int(max_base_cols), max_lag=12, max_total_features=int(max_total_features),
        min_abs_corr=float(min_abs_corr),
        winsor=do_winsor, q_low=q_low, q_high=q_high,
        interp=do_interp, interp_method=interp_method,
        ffill=do_ffill, bfill=do_bfill, standardize=exog_standardize
    )
elif exog_mode == "Manual" and len(exog_cols_user) > 0:
    X_all, exog_lags_map = auto_select_exog_and_lags(
        y_train=train, df_indexed=df, candidate_cols=exog_cols_user,
        max_base_cols=len(exog_cols_user), max_lag=12, max_total_features=int(max_total_features),
        min_abs_corr=float(min_abs_corr),
        winsor=do_winsor, q_low=q_low, q_high=q_high,
        interp=do_interp, interp_method=interp_method,
        ffill=do_ffill, bfill=do_bfill, standardize=exog_standardize
    )

# --- Alinear exógenas al índice completo del objetivo (train+test) y a fin de mes ---
if X_all is not None:
    X_all = ensure_monthly_df(X_all)
    # serie ya está a fin de mes; reindexa exógenas al índice total de la serie
    full_idx = series.index
    X_all = X_all.reindex(full_idx)

    # Utilidades de chequeo seguras (evitan KeyError)
    def _has_issues(df_):
        if df_ is None: return True
        arr = df_.to_numpy(dtype="float64", copy=False)
        return (np.isnan(arr).any()) or (not np.isfinite(arr).all())

    Xt_train_try = X_all.reindex(train.index)
    Xt_test_try  = X_all.reindex(test.index)
    issues_train = _has_issues(Xt_train_try)
    issues_test  = _has_issues(Xt_test_try)

    policy = exog_missing_policy
    if policy == "Permitir extrapolación (interpolar/ffill/bfill)":
        if issues_train or issues_test:
            X_all = (X_all.replace([np.inf, -np.inf], np.nan)
                           .interpolate(method="time")
                           .interpolate(method="linear")
                           .ffill()
                           .bfill())
            # reintenta
            Xt_train_try = X_all.reindex(train.index)
            Xt_test_try  = X_all.reindex(test.index)
            if _has_issues(Xt_train_try) or _has_issues(Xt_test_try):
                X_all = X_all.fillna(X_all.median())
    elif policy == "No extrapolar: degradar a sin exógenas":
        if issues_train or issues_test:
            st.warning("Faltan valores en exógenas (train o test) y la política es **No extrapolar (degradar)**. "
                       "Se continuará **sin exógenas** (Fourier se mantiene si está activado).")
            X_all = None
            exog_lags_map = {}
    elif policy == "No extrapolar: fallar si faltan":
        if issues_train or issues_test:
            st.error("Faltan valores en exógenas (train o test) y la política es **No extrapolar (fallar)**. "
                     "Cambia la política o corrige la cobertura de las exógenas.")
            st.stop()

# Split exógenas (reindex seguro, nunca KeyError)
X_train = X_all.reindex(train.index) if X_all is not None else None
X_test  = X_all.reindex(test.index)  if X_all is not None else None

st.write(f"Observaciones: Train={len(train)}, Test={len(test)}")
if exog_lags_map and X_all is not None:
    st.info(f"Lags seleccionados automáticamente: {exog_lags_map}")
elif exog_mode != "Ninguna" and X_all is None and exog_missing_policy != "No extrapolar: fallar si faltan":
    st.info("No se usarán exógenas (se degradó por política de no extrapolación).")

# Auto-sanar target si hace falta
if robust_mode:
    if train.isna().any() or not np.isfinite(train.values).all():
        train = (train.replace([np.inf, -np.inf], np.nan)
                      .interpolate(method="time")
                      .interpolate(method="linear")
                      .ffill()
                      .bfill())
        if train.isna().any() or not np.isfinite(train.values).all():
            med = float(np.nanmedian(train.values)) if np.isfinite(train.values).any() else 0.0
            train = train.fillna(med)
    if len(train) < int(min_train_months):
        st.warning(f"Train corto ({len(train)} meses). Se reduce la búsqueda (p,q<=1).")
        max_pq = min(max_pq, 1)
else:
    if len(train) < int(min_train_months):
        st.error(f"Train demasiado corto (<{int(min_train_months)} meses). Ajusta ventana o habilita Modo robusto.")
        st.stop()

# ---- Entrenamiento / selección ----
st.subheader("Búsqueda y selección de modelos")
with st.spinner("Entrenando ARIMA / SARIMA / SARIMAX..."):
    out = grid_search_models(
        train, test,
        seasonal_period=int(seasonal_period),
        max_pq=int(max_pq),
        exog_train=X_train, exog_test=X_test,
        include_fourier=bool(include_fourier), K_fourier=range(K_min, K_max+1),
        mape_threshold=float(mape_thr), enforce_threshold=bool(enforce_thr)
    )

if out["best"] is None or len(out["summary"]) == 0:
    st.error("No fue posible ajustar modelos válidos. Revisa los datos."); st.stop()

st.markdown("### Ranking de modelos")
display_cols = ["model","order","seasonal_order","exog","aic","mape","jb_p","lb_p","arch_p","passes_all","meets_thresh"]
summary_df = out["summary"].copy()
for c in display_cols:
    if c not in summary_df.columns: summary_df[c] = np.nan
st.dataframe(summary_df[display_cols].style.format({"aic":"{:.1f}","mape":"{:.2f}%","jb_p":"{:.3f}","lb_p":"{:.3f}","arch_p":"{:.3f}"}))

best = out["best"]
meets = (best.get('mape', 1e9) <= float(mape_thr))
exog_cols_best = best.get("exog_cols", [])
fourier_k  = best.get("fourier_k", None)
st.success(f"**Mejor modelo:** {best['model']} | order={best['order']} | seasonal={best['seasonal_order']} "
           f"| exog_cols={exog_cols_best} | fourier_k={fourier_k} | MAPE={best['mape']:.2f}% "
           f"| Cumple MAPE ≤ {float(mape_thr):.2f}%: {'Sí' if meets else 'No'}")
if best.get("_note"): st.warning(best["_note"])

# Pronóstico (refit de respaldo si hace falta)
def forecast_best(best, train, test, seasonal_period, exog_train, exog_test):
    order = tuple(best["order"]); seasonal_order = tuple(best["seasonal_order"])
    Xtr = exog_train if best.get("exog_cols") else None
    Xte = exog_test  if best.get("exog_cols") else None
    k = best.get("fourier_k", None)
    if k is not None:
        ft_train = fourier_terms(train.index, period=int(seasonal_period), K=int(k))
        ft_test  = fourier_terms(test.index,  period=int(seasonal_period), K=int(k))
        Xtr = ft_train if Xtr is None else pd.concat([Xtr, ft_train], axis=1)
        Xte = ft_test  if Xte is None else pd.concat([Xte, ft_test], axis=1)
    res_tmp = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                      exog=Xtr, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = res_tmp.get_forecast(steps=len(test), exog=Xte).predicted_mean
    fc.index = test.index
    return fc

try:
    fc = best.get("forecast", None)
    if fc is None or len(fc) != len(test):
        fc = forecast_best(best, train, test, int(seasonal_period), X_train, X_test)
except Exception:
    fc = forecast_best(best, train, test, int(seasonal_period), X_train, X_test)

st.markdown("### Pronóstico vs. Real (ventana de evaluación)")
plot_series(train, test, fc, "Pronóstico sobre la muestra de evaluación")

st.markdown("### Diagnósticos del mejor modelo")
st.write({
    "Jarque–Bera p-value (normalidad)": round(float(best["jb_p"]), 4),
    "Ljung–Box p-value (no autocorrelación)": round(float(best["lb_p"]), 4),
    "ARCH p-value (no heterocedasticidad)": round(float(best["arch_p"]), 4),
})

resid_best = refit_and_get_resid(best, train, int(seasonal_period), X_train)
plot_residuals(resid_best, title_prefix=f"{best['model']}")


# === Pruebas de normalidad estilo EViews ===
st.markdown("### Pruebas de Normalidad de los Residuales (estilo EViews)")

resid_best = resid_best.dropna()

# Estadísticos descriptivos
mean_resid = np.mean(resid_best)
std_resid = np.std(resid_best, ddof=1)
skew_resid = stats.skew(resid_best)
kurt_resid = stats.kurtosis(resid_best, fisher=False)
jb_stat, jb_p = stats.jarque_bera(resid_best)

# Tabla tipo EViews
df_norm = pd.DataFrame({
    "Estadístico": [
        "Media de los residuales",
        "Desviación estándar",
        "Asimetría (Skewness)",
        "Curtosis (Kurtosis)",
        "Jarque–Bera",
        "Probabilidad (JB)"
    ],
    "Valor": [
        f"{mean_resid:.6f}",
        f"{std_resid:.6f}",
        f"{skew_resid:.6f}",
        f"{kurt_resid:.6f}",
        f"{jb_stat:.6f}",
        f"{jb_p:.6f}"
    ]
})
st.table(df_norm)

# Interpretación automática
if jb_p > 0.05:
    st.success("✅ Los residuales siguen una distribución normal (no se rechaza H₀ de normalidad).")
else:
    st.warning("⚠️ Los residuales **no** siguen una distribución normal (se rechaza H₀ de normalidad).")

# Histograma + Curva Normal
fig, ax = plt.subplots(figsize=(7,4))
ax.hist(resid_best, bins=20, color="skyblue", edgecolor="black", alpha=0.7, density=True)
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mean_resid, std_resid)
ax.plot(x, p, 'r', linewidth=2)
ax.set_title("Histograma de los Residuales con Curva Normal (EViews Style)")
ax.set_xlabel("Residuales"); ax.set_ylabel("Densidad")
st.pyplot(fig)

# Q–Q Plot (EViews style)
fig = plt.figure(figsize=(6,4))
stats.probplot(resid_best, dist="norm", plot=plt)
plt.title("Q–Q Plot de los Residuales (EViews Style)")
st.pyplot(fig)


with st.expander("SUMMARY del mejor modelo (statsmodels)", expanded=False):
    try:
        res_best = fit_best_model_for_summary(best, train, int(seasonal_period), X_train)
        st.text(res_best.summary().as_text())
    except Exception as e:
        st.warning(f"No se pudo generar el SUMMARY del mejor modelo: {e}")

# ---- Exportar ----
st.markdown("### Exportar resultados")
meets_json = (best.get('mape', 1e9) <= float(mape_thr))
export = {
    "best_model": {
        "model": best["model"],
        "order": tuple(best["order"]),
        "seasonal_order": tuple(best["seasonal_order"]),
        "exog_cols": exog_cols_best,
        "fourier_k": int(fourier_k) if fourier_k is not None else None,
        "aic": float(best["aic"]),
        "mape_eval_%": float(best["mape"]),
        "meets_mape_threshold": bool(meets_json),
        "note": best.get("_note"),
        "diagnostics_pvalues": {"jarque_bera": float(best["jb_p"]),
                                "ljung_box": float(best["lb_p"]),
                                "arch": float(best["arch_p"])}
    },
    "evaluation_window": {"test_start": str(test.index.min()), "test_end": str(test.index.max())},
    "exog_build": {
        "mode": exog_mode,
        "selected_cols": (list(exog_lags_map.keys()) if exog_lags_map else (exog_cols_user if exog_mode=='Manual' else [])),
        "selected_lags_map": exog_lags_map,
        "standardize": bool(exog_standardize),
        "include_fourier": bool(include_fourier),
        "fourier_range": [int(K_min), int(K_max)],
        "missing_policy": exog_missing_policy
    }
}

json_bytes = io.BytesIO()
json_bytes.write(pd.Series(export).to_json().encode("utf-8")); json_bytes.seek(0)
st.download_button("Descargar resumen JSON", data=json_bytes, file_name="soya_model_summary.json", mime="application/json")

csv_bytes = io.BytesIO()
pd.DataFrame({"real": test, "forecast": fc}).to_csv(csv_bytes, index=True); csv_bytes.seek(0)
st.download_button("Descargar pronóstico (CSV)", data=csv_bytes, file_name="forecast_eval.csv", mime="text/csv")
