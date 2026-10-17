import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from scipy.signal import find_peaks
from datetime import datetime, timezone, timedelta, date, time as dt_time
import time

# ----------------- Seiten-Konfiguration -----------------
st.set_page_config(
    page_title="Hurst-Ehlers Pro + Replay Prüfzeitpunkt",
    page_icon="⏳",
    layout="wide"
)

st.title("⏳ Hurst-Ehlers Pro: Zeitreise-Prüfmodus, Top-Down Trichter & FLD")
st.caption("Setze einen beliebigen historischen Prüfzeitpunkt (z. B. vor 4 Tagen) und vergleiche die damalige Zukunftsprojektion direkt mit dem echten Kursverlauf bis heute.")

# US-kompatible Datenquelle (Kraken)
@st.cache_resource
def get_exchange():
    return ccxt.kraken({'enableRateLimit': True})

exchange = get_exchange()

TF_CONFIG = {
    "15m": {"fetch_tf": "15m", "resample": None, "limit": 720, "tv": "15",  "hours": 0.25},
    "1h":  {"fetch_tf": "1h",  "resample": None, "limit": 720, "tv": "60",  "hours": 1.0},
    "4h":  {"fetch_tf": "4h",  "resample": None, "limit": 720, "tv": "240", "hours": 4.0},
    "1d":  {"fetch_tf": "1d",  "resample": None, "limit": 720, "tv": "D",   "hours": 24.0},
    "2d":  {"fetch_tf": "1d",  "resample": "2D", "limit": 720, "tv": "2D",  "hours": 48.0},
    "3d":  {"fetch_tf": "1d",  "resample": "3D", "limit": 720, "tv": "3D",  "hours": 72.0},
    "4d":  {"fetch_tf": "1d",  "resample": "4D", "limit": 720, "tv": "4D",  "hours": 96.0},
    "8d":  {"fetch_tf": "1d",  "resample": "8D", "limit": 720, "tv": "8D",  "hours": 192.0},
    "1w":  {"fetch_tf": "1w",  "resample": None, "limit": 500, "tv": "W",   "hours": 168.0}
}

CASCADE_PRESETS = {
    "Wochen-Trichter (1w ➔ 1d ➔ 4h)": ("1w", "1d", "4h"),
    "Mehr-Tages-Trichter (4d ➔ 1d ➔ 4h)": ("4d", "1d", "4h"),
    "Swing-Trichter (2d ➔ 4h ➔ 1h)": ("2d", "4h", "1h"),
    "Intraday-Trichter (1d ➔ 4h ➔ 1h)": ("1d", "4h", "1h"),
    "Daytrading-Trichter (4h ➔ 1h ➔ 15m)": ("4h", "1h", "15m")
}

ALL_COINS = [
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "NEAR",
    "SUI", "APT", "LTC", "BCH", "UNI", "ATOM", "ARB", "OP", "FIL", "INJ",
    "TIA", "RUNE", "AAVE", "SEI", "RENDER", "ICP", "PEPE"
]

# ----------------- Sidebar / Parameter -----------------
with st.sidebar:
    st.header("⏳ 1. Prüfzeitpunkt (Replay)")
    eval_mode = st.radio(
        "Berechnungs-Stichtag wählen:",
        options=[
            "🔴 Live (Aktuelle Echtzeit)",
            "⏪ Vor X Tagen / Stunden",
            "📅 Exaktes Datum & Uhrzeit"
        ],
        index=1,
        help="Legt fest, zu welchem Zeitpunkt die Prognose berechnet wird. Alle Daten NACH diesem Zeitpunkt werden für das Modell ausgeblendet und dienen als echter Realitäts-Check."
    )
    
    now_utc = datetime.now(timezone.utc)
    if eval_mode == "🔴 Live (Aktuelle Echtzeit)":
        cutoff_dt = now_utc
        is_replay_mode = False
    elif eval_mode == "⏪ Vor X Tagen / Stunden":
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            days_ago = st.number_input("Tage zurück", min_value=0, max_value=120, value=4, step=1)
        with col_r2:
            hours_ago = st.number_input("Stunden zurück", min_value=0, max_value=23, value=0, step=1)
        cutoff_dt = now_utc - timedelta(days=int(days_ago), hours=int(hours_ago))
        is_replay_mode = (days_ago > 0 or hours_ago > 0)
    else:
        default_d = (now_utc - timedelta(days=4)).date()
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            custom_d = st.date_input("Prüf-Datum", value=default_d)
        with col_d2:
            custom_t = st.time_input("Uhrzeit (UTC)", value=dt_time(0, 0), step=3600)
        cutoff_dt = pd.to_datetime(datetime.combine(custom_d, custom_t)).tz_localize('UTC')
        is_replay_mode = True

    st.caption(f"📌 **Aktiver Stichtag:** `{cutoff_dt.strftime('%d.%m.%Y %H:%M UTC')}`")

    st.markdown("---")
    st.header("⚙️ 2. Markt & Zeiteinheit")
    selected_coin = st.selectbox("Einzel-Analyse Asset", options=ALL_COINS, index=0)
    
    tf_options = list(TF_CONFIG.keys())
    tf_choice = st.selectbox("Standard-Timeframe (Hauptchart)", options=tf_options, index=tf_options.index("1d"))
    lookback_bars = st.slider(
        "Historischer Analysebereich (Kerzen)",
        min_value=80, max_value=400, value=200, step=20
    )
    
    st.markdown("---")
    st.subheader("📐 Hurst FLD Einstellungen")
    show_fld = st.checkbox("⚡ Hurst FLD & Kursziel anzeigen", value=True)
    fld_smooth = st.slider("FLD Glättung (EMA-Periode)", min_value=1, max_value=5, value=2, step=1)
    
    st.markdown("---")
    st.subheader("🔭 Top-Down Trichter Kaskade")
    cascade_choice = st.selectbox(
        "Kaskaden-Stufen (Makro ➔ Meso ➔ Mikro)",
        options=list(CASCADE_PRESETS.keys()),
        index=0
    )
    
    st.markdown("---")
    st.subheader("🤖 Auto-Optimizer & Tuning")
    auto_optimize = st.checkbox(
        "⚡ Auto-Optimizer aktivieren",
        value=True,
        help="Kalibriert das Modell automatisch auf einem Walk-Forward-Fenster VOR dem gewählten Prüfzeitpunkt."
    )
    backtest_shift = st.slider(
        "Internes Optimizer-Fenster (Kerzen)",
        min_value=10, max_value=50, value=20, step=5
    )
    
    recency_focus_manual = st.slider(
        "Gegenwarts-Fokus (WLS)", 
        min_value=0.0, max_value=5.0, value=2.5, step=0.5,
        disabled=auto_optimize
    )
    trend_degree_manual = st.selectbox(
        "Basis-Trend Modell", 
        options=[1, 2, 3], 
        index=1,
        format_func=lambda x: "Linear (Gerade)" if x == 1 else f"Polynom ({x}. Grades)",
        disabled=auto_optimize
    )
    num_components_manual = st.slider(
        "Anzahl kombinierter Zyklen",
        min_value=2, max_value=8, value=4, step=1,
        disabled=auto_optimize
    )
    anchor_to_last = st.checkbox("Projektion am Stichtags-Kurs verankern", value=True)
    
    st.markdown("---")
    st.subheader("📡 Ehlers & Horizont")
    min_period = st.number_input("Min. Zykluslänge (Kerzen)", min_value=4, max_value=30, value=6, step=1)
    max_period = st.number_input("Max. Zykluslänge (Kerzen)", min_value=25, max_value=150, value=80, step=5)
    forecast_bars = st.slider("Zukunfts-Projektion ab Stichtag (Kerzen)", min_value=10, max_value=100, value=30, step=5)
    
    st.button("🔄 Analyse aktualisieren", use_container_width=True)

# ----------------- Daten abrufen -----------------
@st.cache_data(ttl=180)
def fetch_candles(coin, tf_key):
    cfg = TF_CONFIG[tf_key]
    symbol = f"{coin}/USD"
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=cfg["fetch_tf"], limit=cfg["limit"])
    if not ohlcv or len(ohlcv) < 50:
        return None
    
    cols = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.DataFrame(ohlcv, columns=cols)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    
    if cfg["resample"]:
        df = df.resample(cfg["resample"]).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
    return df

# ----------------- 1. Multi-Peak Ehlers Spektralanalyse -----------------
def ehlers_multi_peak_spectrum(prices, min_p, max_p, max_comp=8):
    detrended = np.diff(np.log(prices))
    n = len(detrended)
    
    eff_max_p = min(max_p, max(min_p + 2, n // 3 - 1))
    periods = np.arange(min_p, eff_max_p + 1)
    scores = []
    
    for p in periods:
        if n < 3 * p:
            scores.append(0.0)
            continue
        seg1 = detrended[-p:]
        seg2 = detrended[-2*p : -p]
        seg3 = detrended[-3*p : -2*p]
        
        r1 = 0.0
        if np.std(seg1) > 1e-8 and np.std(seg2) > 1e-8:
            r1 = np.corrcoef(seg1, seg2)[0, 1]
            
        r2 = 0.0
        if np.std(seg2) > 1e-8 and np.std(seg3) > 1e-8:
            r2 = np.corrcoef(seg2, seg3)[0, 1]
        
        combined_r = max(0.0, 0.7 * r1 + 0.3 * r2)
        scores.append(combined_r)
        
    score_arr = np.array(scores)
    peaks, _ = find_peaks(score_arr, distance=3)
    
    if len(peaks) > 0:
        sorted_peaks = peaks[np.argsort(score_arr[peaks])[::-1]]
        candidate_p = [float(periods[idx]) for idx in sorted_peaks[:max_comp]]
    elif len(score_arr) > 0:
        best_idx = int(np.argmax(score_arr))
        candidate_p = [float(periods[best_idx])]
    else:
        candidate_p = [float(min_p)]
        
    dom_period = int(candidate_p[0])
    dom_conf = float(np.max(score_arr)) * 100.0 if len(score_arr) > 0 else 0.0
    
    while len(candidate_p) < max_comp:
        harm = max(4.0, round(candidate_p[0] / (len(candidate_p) + 1), 1))
        if harm not in candidate_p:
            candidate_p.append(harm)
        else:
            candidate_p.append(max(4.0, candidate_p[-1] * 0.75))
            
    return dom_period, dom_conf, candidate_p, periods, score_arr

# ----------------- 2. Hurst FLD Berechnung -----------------
def compute_hurst_fld(df_slice, dom_period, smooth_len=2):
    closes = df_slice['Close'].values
    highs = df_slice['High'].values
    lows = df_slice['Low'].values
    n = len(closes)
    
    if smooth_len > 1:
        src = pd.Series(closes).ewm(span=smooth_len, adjust=False).mean().values
    else:
        src = closes.copy()
        
    lag = max(2, int(round(dom_period / 2.0)))
    
    fld_hist = np.full(n, np.nan)
    if n > lag:
        fld_hist[lag:] = src[:-lag]
        
    fld_future = src[-lag:]
    time_step = df_slice.index[-1] - df_slice.index[-2]
    fld_future_times = [df_slice.index[-1] + time_step * (k + 1) for k in range(lag)]
    
    current_fld_level = float(fld_hist[-1]) if not np.isnan(fld_hist[-1]) else float(closes[-1])
    last_close = float(closes[-1])
    
    above_fld = last_close > current_fld_level
    last_cross_idx = None
    last_cross_type = None
    target_price = None
    cross_price = None
    
    for i in range(lag + 1, n):
        if np.isnan(fld_hist[i - 1]) or np.isnan(fld_hist[i]):
            continue
        if closes[i - 1] <= fld_hist[i - 1] and closes[i] > fld_hist[i]:
            last_cross_idx = i
            last_cross_type = "BULL"
            cross_price = float(fld_hist[i])
            win_start = max(0, i - dom_period)
            recent_low = float(np.min(lows[win_start : i + 1]))
            target_price = cross_price + (cross_price - recent_low)
        elif closes[i - 1] >= fld_hist[i - 1] and closes[i] < fld_hist[i]:
            last_cross_idx = i
            last_cross_type = "BEAR"
            cross_price = float(fld_hist[i])
            win_start = max(0, i - dom_period)
            recent_high = float(np.max(highs[win_start : i + 1]))
            target_price = cross_price - (recent_high - cross_price)
            
    dist_pct = ((last_close - current_fld_level) / current_fld_level) * 100.0
    
    if above_fld:
        status_str = f"🟢 Kurs über FLD (+{dist_pct:.2f}%) | Short-Trigger < {current_fld_level:,.4g}"
    else:
        status_str = f"🔴 Kurs unter FLD ({dist_pct:.2f}%) | Long-Trigger > {current_fld_level:,.4g}"
        
    return {
        "lag": lag,
        "fld_hist": fld_hist,
        "fld_future": fld_future,
        "fld_future_times": fld_future_times,
        "current_fld": current_fld_level,
        "above_fld": above_fld,
        "status_str": status_str,
        "last_cross_idx": last_cross_idx,
        "last_cross_type": last_cross_type,
        "cross_price": cross_price,
        "target_price": target_price
    }

# ----------------- 3. Gewichtetes Simultanes Fitting (WLS) -----------------
def fit_high_precision_model(prices, cycle_periods, forecast_n, weight_strength, poly_deg, anchor_last):
    N = len(prices)
    x = np.arange(N)
    total_len = N + forecast_n
    x_full = np.arange(total_len)
    
    if weight_strength > 0:
        weights = np.exp(np.linspace(-weight_strength, 0, N))
    else:
        weights = np.ones(N)
        
    poly_coefs = np.polyfit(x, prices, deg=poly_deg, w=weights)
    trend_hist = np.polyval(poly_coefs, x)
    
    last_val = trend_hist[-1]
    last_slope = trend_hist[-1] - trend_hist[-2]
    trend_future = np.array([last_val + last_slope * (k + 1) * 0.7 for k in range(forecast_n)])
    trend_full = np.concatenate([trend_hist, trend_future])
    
    detrended = prices - trend_hist
    
    basis_cols = []
    for p in cycle_periods:
        w = 2.0 * np.pi / p
        basis_cols.append(np.cos(w * x))
        basis_cols.append(np.sin(w * x))
        
    A_mat = np.column_stack(basis_cols)
    
    W_sqrt = np.sqrt(weights)[:, None]
    A_weighted = A_mat * W_sqrt
    y_weighted = detrended * np.sqrt(weights)
    
    coefs, _, _, _ = np.linalg.lstsq(A_weighted, y_weighted, rcond=None)
    
    composite_wave = np.zeros(total_len)
    wave_dict = {}
    wave_specs = []
    
    for idx, p in enumerate(cycle_periods):
        c_cos = coefs[2 * idx]
        c_sin = coefs[2 * idx + 1]
        amp = float(np.sqrt(c_cos**2 + c_sin**2))
        phase = float(np.arctan2(c_sin, c_cos))
        
        w = 2.0 * np.pi / p
        wave_full = amp * np.cos(w * x_full - phase)
        
        if idx == 0:
            label = f"⭐ Hauptzyklus ({p:.1f} Bars)"
        else:
            label = f"Zyklus #{idx+1} ({p:.1f} Bars)"
            
        wave_dict[label] = wave_full
        composite_wave += wave_full
        
        wave_specs.append({
            "period": float(p),
            "amp": amp,
            "phase": phase,
            "w": float(w),
            "label": f"{int(round(p))}b"
        })
        
    projected_price = trend_full + composite_wave
    
    if anchor_last:
        gap = prices[-1] - projected_price[N - 1]
        decay = np.exp(-np.arange(forecast_n) / (cycle_periods[0] * 0.5))
        projected_price[N:] += gap * decay
        projected_price[:N] += gap * np.exp(-np.arange(N)[::-1] / 15.0)
        
    recent_len = min(50, N)
    corr_matrix = np.corrcoef(prices[-recent_len:], projected_price[N - recent_len : N])
    fit_r2 = corr_matrix[0, 1] * 100.0
    
    return projected_price, wave_dict, max(0.0, float(fit_r2)), wave_specs, composite_wave

# ----------------- 4. Treiber- & Synchronisations-Analyse -----------------
def analyze_turning_point_drivers(bar_idx_full, is_peak, wave_specs):
    sync_waves = []
    opposing_waves = []
    best_trigger_label = "—"
    max_curvature = -1.0
    
    aligned_energy = 0.0
    total_energy = 0.0
    
    for spec in wave_specs:
        p = spec["period"]
        amp = spec["amp"]
        w = spec["w"]
        phase = spec["phase"]
        
        angle = w * bar_idx_full - phase
        cos_val = float(np.cos(angle))
        alignment = cos_val if is_peak else -cos_val
        curvature_force = amp * (w ** 2)
        
        total_energy += amp
        if alignment > 0:
            aligned_energy += amp * alignment
            if alignment * curvature_force > max_curvature:
                max_curvature = alignment * curvature_force
                best_trigger_label = f"Zyklus {int(round(p))} Bars ({int(round(alignment * 100))}% Phase)"
                
        if alignment >= 0.5:
            pct_align = int(round(alignment * 100))
            sync_waves.append((alignment * amp, f"{int(round(p))} Bars ({pct_align}%)"))
        elif alignment <= -0.4:
            opposing_waves.append(f"{int(round(p))}b")
            
    sync_waves.sort(key=lambda x: x[0], reverse=True)
    sync_str = ", ".join([item[1] for item in sync_waves]) if sync_waves else "Nur schwache Teilwellen"
    opp_str = ", ".join(opposing_waves) if opposing_waves else "Keine (Freie Bahn)"
    
    sync_score = round((aligned_energy / total_energy) * 100.0, 1) if total_energy > 0 else 0.0
    sync_count_str = f"{len(sync_waves)} / {len(wave_specs)} Wellen"
    
    return {
        "sync_count": sync_count_str,
        "sync_score": sync_score,
        "trigger": best_trigger_label,
        "drivers": sync_str,
        "opposing": opp_str
    }

# ----------------- 5. Walk-Forward Grid-Search Auto-Optimizer -----------------
def run_walk_forward_optimizer(bt_train, bt_actual, all_bt_peaks, shift_n, anchor_last):
    cycle_counts = [2, 3, 4, 5, 6, 7]
    wls_focus_list = [0.5, 1.5, 2.5, 3.5, 4.5]
    poly_degrees = [1, 2]
    
    actual_diff = np.sign(np.diff(bt_actual))
    results = []
    
    for n_comp in cycle_counts:
        sub_periods = all_bt_peaks[:n_comp]
        for w_foc in wls_focus_list:
            for p_deg in poly_degrees:
                proj_full, _, _, _, _ = fit_high_precision_model(
                    bt_train, sub_periods, shift_n, w_foc, p_deg, anchor_last
                )
                fc_curve = proj_full[len(bt_train):]
                
                corr = 0.0
                if np.std(bt_actual) > 1e-8 and np.std(fc_curve) > 1e-8:
                    corr = float(np.corrcoef(bt_actual, fc_curve)[0, 1]) * 100.0
                
                pred_diff = np.sign(np.diff(fc_curve))
                dir_acc = float(np.mean(actual_diff == pred_diff)) * 100.0
                
                combined_score = 0.5 * dir_acc + 0.5 * max(0.0, corr)
                
                results.append({
                    "Zyklen": n_comp,
                    "WLS-Fokus": w_foc,
                    "Trend-Grad": p_deg,
                    "Trend-Typ": "Linear (1)" if p_deg == 1 else "Polynom (2)",
                    "Richtung (%)": round(dir_acc, 1),
                    "Korrelation (%)": round(corr, 1),
                    "Gesamt-Score": round(combined_score, 1)
                })
                
    res_df = pd.DataFrame(results).sort_values(by="Gesamt-Score", ascending=False).reset_index(drop=True)
    best_row = res_df.iloc[0]
    return best_row, res_df

# ----------------- 6. Top-Down Trichter mit Prüfzeitpunkt-Cutoff -----------------
def compute_tf_continuous_projection(coin, tf_key, cutoff_time, min_p, max_p, n_comp, w_foc, p_deg, smooth_fld=2):
    df_full = fetch_candles(coin, tf_key)
    if df_full is None:
        return None
        
    # Schneide Kerzen am gewählten Prüfzeitpunkt ab
    df_tf = df_full[df_full.index <= cutoff_time].copy()
    df_after = df_full[df_full.index > cutoff_time].copy()
    
    if len(df_tf) < 55:
        return None
        
    sub_df = df_tf.iloc[-min(len(df_tf), 220):].copy()
    prices_tf = sub_df['Close'].values
    n_tf = len(prices_tf)
    
    dom_p, dom_c, peaks_tf, _, _ = ehlers_multi_peak_spectrum(prices_tf, min_p, max_p, max_comp=8)
    sel_p = peaks_tf[:n_comp]
    
    _, _, _, specs_tf, comp_wave_tf = fit_high_precision_model(
        prices_tf, sel_p, 10, w_foc, p_deg, True
    )
    
    fld_info = compute_hurst_fld(sub_df, dom_p, smooth_len=smooth_fld)
    
    t_last = sub_df.index[-1]
    dt_bar = sub_df.index[-1] - sub_df.index[-2]
    
    return {
        "df": sub_df,
        "df_after": df_after,
        "N": n_tf,
        "t_last": t_last,
        "dt_bar": dt_bar,
        "dom_p": dom_p,
        "dom_conf": dom_c,
        "specs": specs_tf,
        "fld_info": fld_info
    }

def evaluate_wave_normalized_at_timestamps(tf_model, target_timestamps):
    N = tf_model["N"]
    t_last = tf_model["t_last"]
    dt_sec = tf_model["dt_bar"].total_seconds()
    specs = tf_model["specs"]
    
    bar_offsets = np.array([(ts - t_last).total_seconds() / dt_sec for ts in target_timestamps])
    x_cont = (N - 1) + bar_offsets
    
    wave_vals = np.zeros(len(target_timestamps))
    total_amp = 0.0
    
    for sp in specs:
        w = sp["w"]
        amp = sp["amp"]
        phase = sp["phase"]
        wave_vals += amp * np.cos(w * x_cont - phase)
        total_amp += amp
        
    if total_amp > 1e-8:
        norm_wave = (wave_vals / total_amp) * 100.0
    else:
        norm_wave = wave_vals
        
    return norm_wave

# ----------------- Einzel-Asset Hauptausführung mit Cutoff -----------------
with st.spinner(f"Berechne Zyklusmodell zum Prüfzeitpunkt {cutoff_dt.strftime('%d.%m.%Y %H:%M')} für {selected_coin}..."):
    df_raw_full = fetch_candles(selected_coin, tf_choice)

if df_raw_full is None or len(df_raw_full) < 80:
    st.error("Nicht genügend Kurshistorie verfügbar.")
    st.stop()

# Daten am Prüfzeitpunkt in "Damals bekannt (Vor Stichtag)" und "Reale Zukunft (Nach Stichtag)" teilen
df_up_to_cutoff = df_raw_full[df_raw_full.index <= cutoff_dt].copy()
df_real_after_cutoff = df_raw_full[df_raw_full.index > cutoff_dt].copy()

min_required = lookback_bars + backtest_shift + 10
if len(df_up_to_cutoff) < min_required:
    st.warning("Für diesen frühen Prüfzeitpunkt reicht der eingestellte Lookback nicht ganz aus – Lookback wird automatisch angepasst.")
    lookback_bars = max(60, len(df_up_to_cutoff) - backtest_shift - 5)
    if lookback_bars < 50:
        st.error("Bitte wähle einen etwas späteren Prüfzeitpunkt (nicht genügend historische Kerzen vor dem Datum).")
        st.stop()

# Interner Auto-Optimizer (arbeitet nur mit Daten VOR dem Prüfzeitpunkt!)
df_bt_train = df_up_to_cutoff.iloc[-(lookback_bars + backtest_shift) : -backtest_shift].copy()
df_bt_actual = df_up_to_cutoff.iloc[-backtest_shift:].copy()
bt_prices_train = df_bt_train['Close'].values
bt_prices_actual = df_bt_actual['Close'].values

bt_dom_p, _, bt_all_peaks, _, _ = ehlers_multi_peak_spectrum(
    bt_prices_train, min_period, max_period, max_comp=8
)

opt_df = None
if auto_optimize:
    best_cfg, opt_df = run_walk_forward_optimizer(
        bt_prices_train, bt_prices_actual, bt_all_peaks, backtest_shift, anchor_to_last
    )
    active_num_comp = int(best_cfg["Zyklen"])
    active_wls_focus = float(best_cfg["WLS-Fokus"])
    active_trend_deg = int(best_cfg["Trend-Grad"])
else:
    active_num_comp = int(num_components_manual)
    active_wls_focus = float(recency_focus_manual)
    active_trend_deg = int(trend_degree_manual)

# Modell-Berechnung exakt am Prüfzeitpunkt (Stichtag)
df_analysis = df_up_to_cutoff.iloc[-lookback_bars:].copy()
prices = df_analysis['Close'].values
N = len(prices)
effective_cutoff_ts = df_analysis.index[-1]

dom_period, dom_conf, all_live_peaks, spec_periods, spec_scores = ehlers_multi_peak_spectrum(
    prices, min_period, max_period, max_comp=8
)
active_periods = all_live_peaks[:active_num_comp]

projected_price, wave_dict, recent_fit_quality, live_wave_specs, _ = fit_high_precision_model(
    prices, active_periods, forecast_bars, active_wls_focus, active_trend_deg, anchor_to_last
)

fld_data = compute_hurst_fld(df_analysis, dom_period, smooth_len=fld_smooth)

time_step = df_analysis.index[-1] - df_analysis.index[-2]
future_times = [df_analysis.index[-1] + time_step * (k + 1) for k in range(forecast_bars)]
all_times = list(df_analysis.index) + future_times

future_curve = projected_price[N:]
peaks, _ = find_peaks(future_curve, distance=2)
troughs, _ = find_peaks(-future_curve, distance=2)

# ----------------- Auswertung der Prognose gegen den ECHTEN Kurs nach dem Prüfzeitpunkt -----------------
n_real_after = len(df_real_after_cutoff)
n_overlap = min(forecast_bars, n_real_after)

replay_dir_acc = None
replay_corr = None
real_after_peaks = np.array([])
real_after_troughs = np.array([])

if n_overlap >= 2:
    real_overlap_prices = df_real_after_cutoff['Close'].values[:n_overlap]
    pred_overlap_prices = future_curve[:n_overlap]
    
    if np.std(real_overlap_prices) > 1e-8 and np.std(pred_overlap_prices) > 1e-8:
        replay_corr = float(np.corrcoef(real_overlap_prices, pred_overlap_prices)[0, 1]) * 100.0
    else:
        replay_corr = 0.0
        
    r_diff = np.sign(np.diff(real_overlap_prices))
    p_diff = np.sign(np.diff(pred_overlap_prices))
    replay_dir_acc = float(np.mean(r_diff == p_diff)) * 100.0
    
    real_after_peaks, _ = find_peaks(df_real_after_cutoff['High'].values, distance=2)
    real_after_troughs, _ = find_peaks(-df_real_after_cutoff['Low'].values, distance=2)

turning_points = []
for p in peaks:
    drv = analyze_turning_point_drivers(N + p, is_peak=True, wave_specs=live_wave_specs)
    fld_at_tp = f"< {fld_data['fld_future'][p]:,.4g}" if p < len(fld_data['fld_future']) else f"< {fld_data['current_fld']:,.4g} (dyn.)"
    
    # Prüfe, ob dieser prognostizierte Wendepunkt bereits in der eingetretenen Realität liegt
    if p < n_real_after:
        if len(real_after_peaks) > 0:
            min_d = int(np.min(np.abs(real_after_peaks - p)))
            if min_d == 0:
                real_check = "✅ Exakter Volltreffer (0 Bars)"
            elif min_d <= 2:
                real_check = f"✅ Treffer in Toleranz (±{min_d} Bar)"
            else:
                real_check = f"❌ Abweichung ({min_d} Bars)"
        else:
            real_check = "❌ Kein lokales Hoch im Zeitraum"
    else:
        real_check = "⏳ Liegt noch in der Zukunft"
        
    turning_points.append({
        "Typ": "▼ Zyklus-HOCH (Top)",
        "Prognose-Zeitpunkt": future_times[p].strftime("%d.%m.%Y %H:%M"),
        "Ab Stichtag": f"+{p + 1} Bars",
        "Realitäts-Check (Ist)": real_check,
        "FLD Bestätigungs-Trigger": f"Short {fld_at_tp}",
        "Cluster-Stärke (%)": drv["sync_score"],
        "Haupt-Trigger": drv["trigger"],
        "Synchrone Treiber": drv["drivers"],
        "Kerzen entfernt": int(p + 1)
    })

for t in troughs:
    drv = analyze_turning_point_drivers(N + t, is_peak=False, wave_specs=live_wave_specs)
    fld_at_tp = f"> {fld_data['fld_future'][t]:,.4g}" if t < len(fld_data['fld_future']) else f"> {fld_data['current_fld']:,.4g} (dyn.)"
    
    if t < n_real_after:
        if len(real_after_troughs) > 0:
            min_d = int(np.min(np.abs(real_after_troughs - t)))
            if min_d == 0:
                real_check = "✅ Exakter Volltreffer (0 Bars)"
            elif min_d <= 2:
                real_check = f"✅ Treffer in Toleranz (±{min_d} Bar)"
            else:
                real_check = f"❌ Abweichung ({min_d} Bars)"
        else:
            real_check = "❌ Kein lokales Tief im Zeitraum"
    else:
        real_check = "⏳ Liegt noch in der Zukunft"
        
    turning_points.append({
        "Typ": "▲ Zyklus-TIEF (Boden)",
        "Prognose-Zeitpunkt": future_times[t].strftime("%d.%m.%Y %H:%M"),
        "Ab Stichtag": f"+{t + 1} Bars",
        "Realitäts-Check (Ist)": real_check,
        "FLD Bestätigungs-Trigger": f"Long {fld_at_tp}",
        "Cluster-Stärke (%)": drv["sync_score"],
        "Haupt-Trigger": drv["trigger"],
        "Synchrone Treiber": drv["drivers"],
        "Kerzen entfernt": int(t + 1)
    })

# ----------------- Detail-Tabs -----------------
tab_main, tab_funnel, tab1, tab_scan, tab_opt, tab2 = st.tabs([
    "📈 Hauptchart + Prüfzeitpunkt-Check",
    "🔭 Top-Down Trichter (zum Prüfzeitpunkt)",
    "🎯 Wendepunkte & Realitäts-Check",
    "🌐 Multi-Coin Scanner",
    "🏆 Auto-Optimizer",
    "🌊 Einzelwellen & Spektrum"
])

# ================= TAB 1: EINZEL-TIMEFRAME HAUPTCHART + REPLAY CHECK =================
with tab_main:
    if is_replay_mode and n_overlap >= 2:
        st.info(
            f"⏳ **Zeitreise-Prüfmodus aktiv:** Berechnungs-Stichtag ist der **{effective_cutoff_ts.strftime('%d.%m.%Y %H:%M UTC')}** "
            f"(danach sind **{n_real_after} echte Kerzen** bis heute vergangen). "
            f"Die damalige Prognose erzielte über diese {n_overlap} Kerzen eine **Richtungs-Trefferquote von {replay_dir_acc:.1f}%** "
            f"(Kurven-Korrelation: **{replay_corr:.1f}%**)."
        )
    elif is_replay_mode:
        st.info(
            f"⏳ **Zeitreise-Prüfmodus aktiv:** Berechnungs-Stichtag ist der **{effective_cutoff_ts.strftime('%d.%m.%Y %H:%M UTC')}**. "
            f"(Für diesen Timeframe `{tf_choice}` sind seit dem Stichtag erst {n_real_after} abgeschlossene Kerzen vergangen)."
        )
    else:
        st.success(f"🔴 **Live-Echtzeit-Modus:** Prognose ab der aktuellen Kerze ({effective_cutoff_ts.strftime('%d.%m.%Y %H:%M UTC')}).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📌 Prüfzeitpunkt (Stichtag)", f"{effective_cutoff_ts.strftime('%d.%m.%y %H:%M')}", f"{n_real_after} Kerzen seitdem vergangen")
    c2.metric("🎯 Dominanter Zyklus (am Stichtag)", f"{dom_period} Kerzen", f"FLD Vorlauf: +{fld_data['lag']} Bars")
    c3.metric("⚡ FLD-Trigger am Stichtag", f"{fld_data['current_fld']:,.4g} $", "Darüber (Bull)" if fld_data["above_fld"] else "Darunter (Bear)")
    
    if replay_dir_acc is not None:
        c4.metric(
            f"✅ Echte Trefferquote ({n_overlap} Bars)",
            f"{replay_dir_acc:.1f} % Richtung",
            f"Korrelation: {replay_corr:.1f} %"
        )
    else:
        c4.metric("In-Sample Fit (am Stichtag)", f"{recent_fit_quality:.1f} %", f"{active_num_comp} Zyklen | WLS {active_wls_focus}")

    st.subheader(f"📈 {selected_coin} ({tf_choice}) – Vorhersage am Prüfzeitpunkt vs. Echter Kursverlauf bis heute")

    fig = go.Figure()
    
    # 1. Realer Kurs BIS zum Prüfzeitpunkt (dem Modell bekannt)
    fig.add_trace(go.Scatter(
        x=df_analysis.index, y=df_analysis['Close'], mode='lines',
        name='Bekannter Kurs (bis Prüfzeitpunkt)', line=dict(color='#d1d4dc', width=2)
    ))
    
    # 2. Tatsächlicher Kurs NACH dem Prüfzeitpunkt bis heute (zum Überprüfen der Prognose!)
    if n_real_after > 0:
        real_after_x = [df_analysis.index[-1]] + list(df_real_after_cutoff.index)
        real_after_y = [float(df_analysis['Close'].iloc[-1])] + list(df_real_after_cutoff['Close'].values)
        fig.add_trace(go.Scatter(
            x=real_after_x, y=real_after_y, mode='lines+markers',
            name='⚪ Tatsächlicher Kurs NACH Prüfzeitpunkt (Realität bis heute)',
            line=dict(color='#ffffff', width=2.8),
            marker=dict(size=5, color='#ffffff')
        ))
    
    # 3. Historische & zukünftige Hurst FLD Linie (vom Prüfzeitpunkt aus)
    if show_fld:
        fig.add_trace(go.Scatter(
            x=df_analysis.index, y=fld_data["fld_hist"], mode='lines',
            name=f'⚡ Hurst FLD (bis Stichtag)',
            line=dict(color='#ec4899', width=1.6)
        ))
        fld_fut_x = [df_analysis.index[-1]] + fld_data["fld_future_times"]
        fld_fut_y = [fld_data["current_fld"]] + list(fld_data["fld_future"])
        fig.add_trace(go.Scatter(
            x=fld_fut_x, y=fld_fut_y, mode='lines',
            name=f'🔮 Zukünftige FLD (ab Stichtag +{fld_data["lag"]}b)',
            line=dict(color='#ec4899', width=2.2, dash='dash')
        ))
        
        if fld_data["target_price"] is not None:
            is_bull_c = fld_data["last_cross_type"] == "BULL"
            fig.add_hline(
                y=fld_data["target_price"],
                line_width=1.3,
                line_dash="dot",
                line_color="#089981" if is_bull_c else "#f23645",
                annotation_text=f"🎯 Hurst 1:1 FLD-Ziel: {fld_data['target_price']:,.4g}",
                annotation_position="top left"
            )

    # 4. In-Sample Modell-Fit (bis Prüfzeitpunkt)
    fig.add_trace(go.Scatter(
        x=df_analysis.index, y=projected_price[:N], mode='lines',
        name='Hurst-Summenkurve (Fit bis Stichtag)', line=dict(color='#3b82f6', width=1.5, dash='dot')
    ))

    # 5. Damalige Zukunftsprojektion (ab Prüfzeitpunkt)
    proj_x = [df_analysis.index[-1]] + future_times
    proj_y = [projected_price[N - 1]] + list(projected_price[N:])
    fig.add_trace(go.Scatter(
        x=proj_x, y=proj_y, mode='lines',
        name='🔮 Zyklus-Projektion (berechnet am Prüfzeitpunkt)',
        line=dict(color='#089981', width=3)
    ))

    if len(peaks) > 0:
        fig.add_trace(go.Scatter(
            x=[future_times[p] for p in peaks], y=[future_curve[p] for p in peaks],
            mode='markers', name='Prognostiziertes Top ▼',
            marker=dict(color='#f23645', size=11, symbol='triangle-down')
        ))

    if len(troughs) > 0:
        fig.add_trace(go.Scatter(
            x=[future_times[t] for t in troughs], y=[future_curve[t] for t in troughs],
            mode='markers', name='Prognostizierter Boden ▲',
            marker=dict(color='#089981', size=11, symbol='triangle-up')
        ))

    # Vertikale rote Linie am gewählten Prüfzeitpunkt
    fig.add_vline(
        x=effective_cutoff_ts.timestamp() * 1000,
        line_width=2, line_dash="dash", line_color="#f23645"
    )
    fig.add_annotation(
        x=effective_cutoff_ts,
        y=float(prices[-1]),
        text="⏳ PRÜFZEITPUNKT",
        showarrow=True,
        arrowhead=1,
        arrowcolor="#f23645",
        font=dict(color="#f23645", size=12)
    )

    fig.update_layout(
        template="plotly_dark", height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 2: TOP-DOWN TRICHTER ZUM PRÜFZEITPUNKT =================
with tab_funnel:
    tf_macro, tf_meso, tf_micro = CASCADE_PRESETS[cascade_choice]
    st.subheader(
        f"🔭 Top-Down Trichter für {selected_coin} (`{tf_macro}` ➔ `{tf_meso}` ➔ `{tf_micro}`) "
        f"zum Prüfzeitpunkt `{cutoff_dt.strftime('%d.%m.%Y %H:%M UTC')}`"
    )
    st.caption(
        "Auch der Trichter wird exakt zu deinem gewählten Prüfzeitpunkt berechnet. "
        "Wenn dein Prüfzeitpunkt z. B. vor 4 Tagen war, siehst du hier, welchen genauen Tag und welche Stunde der Trichter damals vorhergesagt hat!"
    )
    
    mod_macro = compute_tf_continuous_projection(selected_coin, tf_macro, cutoff_dt, min_period, max_period, active_num_comp, active_wls_focus, active_trend_deg, fld_smooth)
    mod_meso  = compute_tf_continuous_projection(selected_coin, tf_meso,  cutoff_dt, min_period, max_period, active_num_comp, active_wls_focus, active_trend_deg, fld_smooth)
    mod_micro = compute_tf_continuous_projection(selected_coin, tf_micro, cutoff_dt, min_period, max_period, active_num_comp, active_wls_focus, active_trend_deg, fld_smooth)
    
    if mod_macro and mod_meso and mod_micro:
        macro_horizon_bars = 8
        total_hours = int(macro_horizon_bars * TF_CONFIG[tf_macro]["hours"])
        micro_step_hours = TF_CONFIG[tf_micro]["hours"]
        num_micro_steps = min(400, int(total_hours / micro_step_hours))
        
        now_anchor = mod_micro["t_last"]
        dt_micro = mod_micro["dt_bar"]
        
        grid_times = [now_anchor + dt_micro * k for k in range(-30, num_micro_steps + 1)]
        future_start_idx = 30
        
        w_macro_norm = evaluate_wave_normalized_at_timestamps(mod_macro, grid_times)
        w_meso_norm  = evaluate_wave_normalized_at_timestamps(mod_meso,  grid_times)
        w_micro_norm = evaluate_wave_normalized_at_timestamps(mod_micro, grid_times)
        
        mtf_confluence_wave = 0.40 * w_macro_norm + 0.35 * w_meso_norm + 0.25 * w_micro_norm
        
        fut_macro = w_macro_norm[future_start_idx:]
        fut_meso  = w_meso_norm[future_start_idx:]
        fut_micro = w_micro_norm[future_start_idx:]
        fut_times = grid_times[future_start_idx:]
        
        bars_per_macro = max(2, int(TF_CONFIG[tf_macro]["hours"] / micro_step_hours))
        bars_per_meso  = max(1, int(TF_CONFIG[tf_meso]["hours"] / micro_step_hours))
        
        m_peaks, _ = find_peaks(fut_macro, distance=bars_per_macro)
        m_troughs, _ = find_peaks(-fut_macro, distance=bars_per_macro)
        
        funnel_events = []
        pinpoint_markers = []
        
        micro_fld_val = mod_micro["fld_info"]["current_fld"]
        meso_fld_val  = mod_meso["fld_info"]["current_fld"]
        
        def pinpoint_macro_event(m_idx, is_top):
            half_win = max(2, bars_per_macro // 2)
            w_start = max(0, m_idx - half_win)
            w_end   = min(len(fut_times) - 1, m_idx + half_win)
            
            macro_window_str = f"{fut_times[w_start].strftime('%d.%m.')} – {fut_times[w_end].strftime('%d.%m.%Y')}"
            
            meso_slice = fut_meso[w_start : w_end + 1] if is_top else -fut_meso[w_start : w_end + 1]
            best_meso_rel = int(np.argmax(meso_slice))
            best_meso_idx = w_start + best_meso_rel
            
            meso_half = max(1, bars_per_meso)
            m2_start = max(w_start, best_meso_idx - meso_half)
            m2_end   = min(w_end, best_meso_idx + meso_half)
            meso_day_str = f"{fut_times[best_meso_idx].strftime('%d.%m.%Y')} (±1 Tag)"
            
            combined_micro_slice = (
                (0.35 * fut_macro[m2_start:m2_end+1] + 0.35 * fut_meso[m2_start:m2_end+1] + 0.30 * fut_micro[m2_start:m2_end+1])
                if is_top else
                -(0.35 * fut_macro[m2_start:m2_end+1] + 0.35 * fut_meso[m2_start:m2_end+1] + 0.30 * fut_micro[m2_start:m2_end+1])
            )
            best_micro_rel = int(np.argmax(combined_micro_slice))
            pin_idx = m2_start + best_micro_rel
            pin_time = fut_times[pin_idx]
            
            val_mac = fut_macro[pin_idx] if is_top else -fut_macro[pin_idx]
            val_mes = fut_meso[pin_idx]  if is_top else -fut_meso[pin_idx]
            val_mic = fut_micro[pin_idx] if is_top else -fut_micro[pin_idx]
            
            score_mac = max(0.0, (val_mac + 100.0) / 2.0)
            score_mes = max(0.0, (val_mes + 100.0) / 2.0)
            score_mic = max(0.0, (val_mic + 100.0) / 2.0)
            total_conf_score = round(0.40 * score_mac + 0.35 * score_mes + 0.25 * score_mic, 1)
            
            hours_away = round((pin_time - now_anchor).total_seconds() / 3600.0, 1)
            days_away = round(hours_away / 24.0, 1)
            
            pinpoint_markers.append({
                "time": pin_time,
                "y_val": mtf_confluence_wave[future_start_idx + pin_idx],
                "is_top": is_top,
                "label": f"{'▼ TOP' if is_top else '▲ BODEN'} ({pin_time.strftime('%d.%m. %H:%M')})"
            })
            
            op_sym = "<" if is_top else ">"
            fld_trigger_str = f"{tf_micro}-FLD {op_sym} {micro_fld_val:,.4g} | {tf_meso}-FLD {op_sym} {meso_fld_val:,.4g}"
            status_pin = "✅ Bereits vergangen (Prüfbar im Chart)" if pin_time <= now_utc else "🔮 Liegt noch in der Zukunft"
            
            return {
                "Richtung": "▼ SHORT (Zyklus-Top)" if is_top else "▲ LONG (Zyklus-Boden)",
                f"1. Makro-Zone ({tf_macro})": macro_window_str,
                f"2. Meso-Fokus ({tf_meso})": meso_day_str,
                f"3. Exakter {tf_micro} Pinpoint": pin_time.strftime("%d.%m.%Y %H:%M UTC"),
                "Status": status_pin,
                "FLD Preis-Bestätigung": fld_trigger_str,
                "Ab Stichtag": f"+{days_away}d ({int(hours_away)}h)",
                "3-TF Synchronisation": total_conf_score,
                "sort_hours": hours_away
            }
            
        for pk in m_peaks:
            funnel_events.append(pinpoint_macro_event(pk, is_top=True))
        for tr in m_troughs:
            funnel_events.append(pinpoint_macro_event(tr, is_top=False))
            
        if funnel_events:
            funnel_df = pd.DataFrame(funnel_events).sort_values(by="sort_hours").reset_index(drop=True)
            st.dataframe(
                funnel_df.drop(columns=["sort_hours"]),
                column_config={
                    "3-TF Synchronisation": st.column_config.ProgressColumn(
                        "Multi-TF Konfluenz", format="%.1f %%", min_value=0, max_value=100
                    )
                },
                use_container_width=True,
                hide_index=True
            )
            
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=grid_times, y=w_macro_norm, mode='lines', name=f'1. Makro-Welle ({tf_macro})', line=dict(color='#3b82f6', width=2.5)))
        fig_f.add_trace(go.Scatter(x=grid_times, y=w_meso_norm, mode='lines', name=f'2. Meso-Welle ({tf_meso})', line=dict(color='#f59e0b', width=1.8, dash='dot')))
        fig_f.add_trace(go.Scatter(x=grid_times, y=w_micro_norm, mode='lines', name=f'3. Mikro-Welle ({tf_micro})', line=dict(color='#a855f7', width=1.2)))
        fig_f.add_trace(go.Scatter(x=grid_times, y=mtf_confluence_wave, mode='lines', name='⚡ Kombinierte 3-TF Summenwelle', line=dict(color='#089981', width=3)))
        
        for pm in pinpoint_markers:
            fig_f.add_trace(go.Scatter(
                x=[pm["time"]], y=[pm["y_val"]], mode='markers+text',
                name=pm["label"], text=[pm["label"]],
                textposition="top center" if pm["is_top"] else "bottom center",
                marker=dict(color='#f23645' if pm["is_top"] else '#089981', size=14, symbol='star'),
                showlegend=False
            ))
            
        fig_f.add_vline(x=now_anchor.timestamp() * 1000, line_width=2, line_dash="dash", line_color="#f23645")
        fig_f.add_hline(y=0, line_width=1, line_dash="dot", line_color="#4b5563")
        fig_f.update_layout(
            template="plotly_dark", height=480,
            yaxis_title="Normierte Zyklus-Phase (-100% Boden bis +100% Top)",
            yaxis=dict(range=[-125, 125]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_f, use_container_width=True)

# ================= TAB 3: WENDEPUNKTE & REALITÄTS-CHECK =================
with tab1:
    st.markdown(f"##### 🎯 Prognostizierte Wendepunkte ab Stichtag (`{effective_cutoff_ts.strftime('%d.%m.%Y %H:%M')}`) inkl. Realitäts-Check")
    if turning_points:
        df_tp = pd.DataFrame(turning_points).sort_values(by="Kerzen entfernt").drop(columns=["Kerzen entfernt"])
        st.dataframe(
            df_tp,
            column_config={
                "Cluster-Stärke (%)": st.column_config.ProgressColumn(
                    "Cluster-Synchronisation", format="%.1f %%", min_value=0, max_value=100
                )
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Keine klaren Wendepunkte im gewählten Horizont erkannt.")

# ================= TAB 4: MULTI-COIN SCANNER =================
with tab_scan:
    st.markdown(f"##### 🌐 Multi-Coin Scanner zum Prüfzeitpunkt ({tf_choice})")
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        num_scan_coins = st.slider("Anzahl zu scannender Coins", min_value=5, max_value=len(ALL_COINS), value=15, step=5)
    with col_s2:
        st.write("")
        start_scan = st.button("🚀 Alle Coins jetzt scannen & ranken", type="primary", use_container_width=True)
        
    if start_scan:
        scan_list = ALL_COINS[:num_scan_coins]
        scan_rows = []
        prog_text = st.empty()
        prog_bar = st.progress(0)
        tv_tf_code = TF_CONFIG[tf_choice]["tv"]
        
        for idx_c, c_sym in enumerate(scan_list):
            prog_text.text(f"Optimiere Zyklus & prüfe Prognose ({idx_c + 1}/{len(scan_list)}): {c_sym} ({tf_choice}) ...")
            try:
                df_c_full = fetch_candles(c_sym, tf_choice)
                if df_c_full is not None:
                    df_c = df_c_full[df_c_full.index <= cutoff_dt].copy()
                    if len(df_c) >= min_required:
                        c_bt_train = df_c.iloc[-(lookback_bars + backtest_shift) : -backtest_shift]['Close'].values
                        c_bt_actual = df_c.iloc[-backtest_shift:]['Close'].values
                        
                        _, _, c_bt_peaks, _, _ = ehlers_multi_peak_spectrum(c_bt_train, min_period, max_period, max_comp=8)
                        best_c_row, _ = run_walk_forward_optimizer(
                            c_bt_train, c_bt_actual, c_bt_peaks, backtest_shift, anchor_to_last
                        )
                        
                        c_live_df = df_c.iloc[-lookback_bars:]
                        c_live_prices = c_live_df['Close'].values
                        c_dom_p, _, c_live_peaks, _, _ = ehlers_multi_peak_spectrum(
                            c_live_prices, min_period, max_period, max_comp=8
                        )
                        
                        opt_n = int(best_c_row["Zyklen"])
                        opt_w = float(best_c_row["WLS-Fokus"])
                        opt_d = int(best_c_row["Trend-Grad"])
                        
                        c_proj, _, _, c_specs, _ = fit_high_precision_model(
                            c_live_prices, c_live_peaks[:opt_n], forecast_bars, opt_w, opt_d, anchor_to_last
                        )
                        
                        c_fld = compute_hurst_fld(c_live_df, c_dom_p, smooth_len=fld_smooth)
                        
                        c_fut_curve = c_proj[len(c_live_prices):]
                        c_tstep = c_live_df.index[-1] - c_live_df.index[-2]
                        c_ftimes = [c_live_df.index[-1] + c_tstep * (k + 1) for k in range(forecast_bars)]
                        
                        c_pk, _ = find_peaks(c_fut_curve, distance=2)
                        c_tr, _ = find_peaks(-c_fut_curve, distance=2)
                        
                        next_events = []
                        for p_i in c_pk:
                            drv_i = analyze_turning_point_drivers(len(c_live_prices) + p_i, True, c_specs)
                            next_events.append((
                                int(p_i + 1),
                                f"▼ TOP in +{p_i+1}b ({c_ftimes[p_i].strftime('%d.%m.')}) | Sync: {drv_i['sync_score']}%"
                            ))
                        for t_i in c_tr:
                            drv_i = analyze_turning_point_drivers(len(c_live_prices) + t_i, False, c_specs)
                            next_events.append((
                                int(t_i + 1),
                                f"▲ BODEN in +{t_i+1}b ({c_ftimes[t_i].strftime('%d.%m.')}) | Sync: {drv_i['sync_score']}%"
                            ))
                            
                        next_events.sort(key=lambda x: x[0])
                        next_turn_str = next_events[0][1] if next_events else "—"
                        
                        fld_state_short = f"{'🟢 >' if c_fld['above_fld'] else '🔴 <'} {c_fld['current_fld']:,.4g}"
                        tv_link = f"https://www.tradingview.com/chart/?symbol=BYBIT:{c_sym}USDT.P&interval={tv_tf_code}"
                        
                        scan_rows.append({
                            "Symbol": f"{c_sym}USDT.P",
                            "Gesamt-Score": float(best_c_row["Gesamt-Score"]),
                            "Richtung (%)": float(best_c_row["Richtung (%)"]),
                            "Dom. Zyklus": f"{c_dom_p}b (FLD +{c_fld['lag']}b)",
                            "FLD Trigger-Level": fld_state_short,
                            "Nächster Wendepunkt & Sync": next_turn_str,
                            "TradingView": tv_link
                        })
            except Exception:
                pass
            prog_bar.progress((idx_c + 1) / len(scan_list))
            time.sleep(0.03)
            
        prog_text.empty()
        prog_bar.empty()
        
        if scan_rows:
            st.session_state["last_scan_df"] = pd.DataFrame(scan_rows).sort_values(
                by="Gesamt-Score", ascending=False
            ).reset_index(drop=True)
            
    if "last_scan_df" in st.session_state:
        st.dataframe(
            st.session_state["last_scan_df"],
            column_config={
                "Gesamt-Score": st.column_config.ProgressColumn(
                    "Walk-Forward Score", format="%.1f", min_value=0, max_value=100
                ),
                "TradingView": st.column_config.LinkColumn("Chart", display_text="Chart öffnen ↗")
            },
            use_container_width=True,
            hide_index=True
        )

# ================= TAB 5: AUTO-OPTIMIZER RANKING =================
with tab_opt:
    st.markdown(f"##### 🏆 Rangliste der getesteten Parameter für {selected_coin} (am Prüfzeitpunkt)")
    if opt_df is not None:
        st.dataframe(
            opt_df.head(15).drop(columns=["Trend-Grad"]),
            column_config={
                "Gesamt-Score": st.column_config.ProgressColumn("Gesamt-Score", format="%.1f", min_value=0, max_value=100)
            },
            use_container_width=True,
            hide_index=True
        )

# ================= TAB 6: EINZELWELLEN & SPEKTRUM =================
with tab2:
    c_w1, c_w2 = st.columns(2)
    with c_w1:
        st.markdown("##### Isolierte Zykluswellen")
        fig_waves = go.Figure()
        for w_name, w_vals in wave_dict.items():
            fig_waves.add_trace(go.Scatter(x=all_times, y=w_vals, mode='lines', name=w_name))
        fig_waves.add_vline(x=effective_cutoff_ts.timestamp() * 1000, line_dash="dash", line_color="#f23645")
        fig_waves.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_waves, use_container_width=True)
    with c_w2:
        st.markdown("##### Ehlers Autokorrelations-Spektrum")
        spec_df = pd.DataFrame({"Periode": spec_periods, "Score": spec_scores * 100.0})
        active_int = [int(round(p)) for p in active_periods]
        bar_colors = ["#089981" if p in active_int else "#2a2e39" for p in spec_df["Periode"]]
        fig_spec = go.Figure(go.Bar(x=spec_df["Periode"], y=spec_df["Score"], marker_color=bar_colors))
        fig_spec.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_spec, use_container_width=True)
