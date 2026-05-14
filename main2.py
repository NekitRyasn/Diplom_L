#claud
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
from datetime import date, timedelta

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Оценка опционов — Блэк-Шоулс",
    page_icon="📈",
    layout="wide",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.block-container { padding-top: 2rem; }

.result-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 0.5rem 0;
    text-align: center;
}
.result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #8b949e;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-call {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 600;
    color: #3fb950;
}
.result-put {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 600;
    color: #f85149;
}
.vol-display {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #e3b341;
    text-align: center;
}
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #8b949e;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
.ticker-info {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #8b949e;
}
.stButton > button {
    background: #238636;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.04em;
    width: 100%;
}
.stButton > button:hover { background: #2ea043; border: none; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
TICKERS = {
    "Apple (AAPL)":       "AAPL",
    "Microsoft (MSFT)":   "MSFT",
    "Google (GOOGL)":     "GOOGL",
    "Amazon (AMZN)":      "AMZN",
    "Tesla (TSLA)":       "TSLA",
    "NVIDIA (NVDA)":      "NVDA",
    "Meta (META)":        "META",
    "Нефть WTI (CL=F)":  "CL=F",
    "Золото (GC=F)":      "GC=F",
    "S&P 500 (^GSPC)":   "^GSPC",
    "Bitcoin (BTC-USD)":  "BTC-USD",
}


# ─── Helper functions ─────────────────────────────────────────────────────────

def extract_series(data: pd.DataFrame) -> pd.Series:
    """
    Safely extract a 1-D Close price Series from yfinance output.
    Works with both simple ColumnIndex and MultiIndex (field, ticker).
    """
    close = data["Close"]
    # If MultiIndex columns → DataFrame → take first column
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    # Re-create as plain Series to guarantee no leftover MultiIndex
    result = pd.Series(
        close.values.ravel().astype(float),
        index=pd.to_datetime(data.index).tz_localize(None),
        name="Close",
    )
    return result.dropna()


@st.cache_data(ttl=300)
def load_prices(ticker: str) -> pd.Series:
    data = yf.download(ticker, period="30mo", interval="1d",
                       progress=False, auto_adjust=True)
    return extract_series(data)


@st.cache_data(ttl=3600)
def load_risk_free_rate() -> float:
    try:
        tnx = yf.Ticker("^TNX").history(period="1d")
        return float(tnx["Close"].iloc[-1]) / 100
    except Exception:
        return 0.05


def calculate_volatility(prices: pd.Series, N: int) -> float:
    log_returns = np.log(prices / prices.shift(1)).dropna()
    N = min(N, len(log_returns))
    mu  = float(log_returns.mean())
    vol = float(np.sqrt((1.0 / (N - 1)) * np.sum((log_returns[:N] - mu) ** 2)))
    return vol


def black_scholes(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None, None
    d_plus  = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d_minus = d_plus - sigma * np.sqrt(T)
    call = S * norm.cdf(d_plus) - K * np.exp(-r * T) * norm.cdf(d_minus)
    put  = K * np.exp(-r * T) * norm.cdf(-d_minus) - S * norm.cdf(-d_plus)
    return float(call), float(put)


# ─── UI ───────────────────────────────────────────────────────────────────────

st.markdown("# 📊 Оценка опционов")
st.markdown("**Модель Блэка-Шоулса** · Расчёт цен колл и пут опционов")
st.divider()

col_left, col_right = st.columns([1, 1.6], gap="large")

# ════════════════════════════════════════
#  LEFT PANEL — inputs
# ════════════════════════════════════════
with col_left:

    # ── 01 Ticker ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">01 · Выбор актива</p>', unsafe_allow_html=True)
    ticker_label = st.selectbox("Актив", list(TICKERS.keys()), index=0,
                                label_visibility="collapsed")
    ticker = TICKERS[ticker_label]

    load_btn = st.button("⬇  Загрузить данные с Yahoo Finance")

    if load_btn or "prices" not in st.session_state \
            or st.session_state.get("loaded_ticker") != ticker:
        if load_btn or "prices" not in st.session_state:
            with st.spinner("Загрузка данных..."):
                try:
                    p = load_prices(ticker)
                    st.session_state["prices"]        = p
                    st.session_state["loaded_ticker"] = ticker
                    st.session_state["r_default"]     = load_risk_free_rate()
                except Exception as e:
                    st.error(f"Ошибка загрузки: {e}")

    prices: pd.Series = st.session_state.get("prices", pd.Series(dtype=float))

    if not prices.empty:
        last_price = float(prices.iloc[-1])
        st.markdown(
            f'<div class="ticker-info">'
            f'Последняя цена: <b>{last_price:.2f}</b> &nbsp;·&nbsp; '
            f'Данных: {len(prices)} дней &nbsp;·&nbsp; '
            f'с {prices.index[0].date()} по {prices.index[-1].date()}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        last_price = 100.0

    st.markdown("")

    # ── 02 Volatility ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">02 · Волатильность (σ)</p>', unsafe_allow_html=True)
    vol_type = st.radio(
        "Периодичность",
        ["Ежедневная (200 наблюдений)",
         "Еженедельная (50 наблюдений)",
         "Ежемесячная (30 наблюдений)"],
        label_visibility="collapsed",
    )

    sigma_calc = None
    if not prices.empty:
        try:
            if "Ежедневная" in vol_type:
                sigma_calc = calculate_volatility(prices, 200)
            elif "Еженедельная" in vol_type:
                weekly = prices.resample("W").last()
                sigma_calc = calculate_volatility(weekly, 50)
            else:
                monthly = prices.resample("ME").last()
                sigma_calc = calculate_volatility(monthly, 30)
        except Exception as e:
            st.warning(f"Ошибка расчёта волатильности: {e}")

    st.markdown(
        f'<div class="vol-display">σ = {sigma_calc:.6f}</div>'
        if sigma_calc is not None
        else '<div class="vol-display">σ = —</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── 03 Parameters ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">03 · Параметры модели</p>', unsafe_allow_html=True)

    today          = date.today()
    default_expiry = today + timedelta(days=30)
    default_r_pct  = round(st.session_state.get("r_default", 0.05) * 100, 4)

    c1, c2 = st.columns(2)
    with c1:
        expiry_date = st.date_input("Дата экспирации", value=default_expiry,
                                    min_value=today + timedelta(days=1))
    with c2:
        T_days = (expiry_date - today).days
        T_val  = T_days / 365.0
        st.metric("T (лет)", f"{T_val:.4f}", f"{T_days} дней")

    c3, c4 = st.columns(2)
    with c3:
        S = st.number_input("S₀ — цена актива", value=round(last_price, 4),
                            min_value=0.01, format="%.4f")
    with c4:
        K = st.number_input("K — страйк-цена",  value=round(last_price, 4),
                            min_value=0.01, format="%.4f")

    c5, c6 = st.columns(2)
    with c5:
        r_pct = st.number_input("r — ставка (%)", value=default_r_pct,
                                min_value=0.0, max_value=100.0, format="%.4f")
        r = r_pct / 100.0
    with c6:
        sigma_input = st.number_input(
            "σ — волатильность",
            value=round(sigma_calc, 6) if sigma_calc else 0.2,
            min_value=0.0001, max_value=10.0, format="%.6f",
            help="Считается автоматически, но можно ввести вручную",
        )


# ════════════════════════════════════════
#  RIGHT PANEL — results + charts
# ════════════════════════════════════════
with col_right:

    # ── 04 Black-Scholes ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">04 · Результаты модели Блэка-Шоулса</p>',
                unsafe_allow_html=True)

    call_price, put_price = black_scholes(S, K, T_val, r, sigma_input)

    if call_price is not None:
        d_plus  = (np.log(S / K) + (r + 0.5 * sigma_input**2) * T_val) / (sigma_input * np.sqrt(T_val))
        d_minus = d_plus - sigma_input * np.sqrt(T_val)

        st.markdown("**Параметры расчёта:**")
        params_df = pd.DataFrame({
            "Параметр": ["S₀", "K", "T", "r", "σ", "d₊", "d₋"],
            "Значение": [
                f"{S:.4f}", f"{K:.4f}", f"{T_val:.6f}",
                f"{r:.4f} ({r_pct:.2f}%)", f"{sigma_input:.6f}",
                f"{d_plus:.6f}", f"{d_minus:.6f}",
            ],
        })
        st.dataframe(params_df, hide_index=True, use_container_width=True)

        st.markdown("**Цены опционов:**")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown(f"""
<div class="result-card">
    <div class="result-label">Колл-опцион (C)</div>
    <div class="result-call">{call_price:.4f}</div>
    <div class="result-label" style="margin-top:0.5rem">
        C = S·N(d₊) − K·e⁻ʳᵀ·N(d₋)
    </div>
</div>""", unsafe_allow_html=True)
        with rc2:
            st.markdown(f"""
<div class="result-card">
    <div class="result-label">Пут-опцион (P)</div>
    <div class="result-put">{put_price:.4f}</div>
    <div class="result-label" style="margin-top:0.5rem">
        P = K·e⁻ʳᵀ·N(−d₋) − S·N(−d₊)
    </div>
</div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── 05 Greeks ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-header">05 · Греки</p>', unsafe_allow_html=True)

        N_prime        = norm.pdf(d_plus)
        delta_call_val = norm.cdf(d_plus)
        delta_put_val  = norm.cdf(d_plus) - 1
        gamma_val      = N_prime / (S * sigma_input * np.sqrt(T_val))
        vega_val       = S * np.sqrt(T_val) * N_prime
        theta_call_val = (-(S * sigma_input * N_prime) / (2 * np.sqrt(T_val))
                          - r * K * norm.cdf(d_minus) * np.exp(-r * T_val)) / 365
        theta_put_val  = (-(S * sigma_input * N_prime) / (2 * np.sqrt(T_val))
                          + r * K * norm.cdf(-d_minus) * np.exp(-r * T_val)) / 365

        greeks_df = pd.DataFrame({
            "Грек": ["Δ Delta", "Γ Gamma", "ν Vega", "Θ Theta (в день)"],
            "Колл": [f"{delta_call_val:.6f}", f"{gamma_val:.6f}",
                     f"{vega_val:.4f}",       f"{theta_call_val:.6f}"],
            "Пут":  [f"{delta_put_val:.6f}",  f"{gamma_val:.6f}",
                     f"{vega_val:.4f}",        f"{theta_put_val:.6f}"],
        })
        st.dataframe(greeks_df, hide_index=True, use_container_width=True)

        parity = call_price - put_price - (S - K * np.exp(-r * T_val))
        st.caption(f"Паритет колл-пут: C − P − (S − K·e⁻ʳᵀ) = {parity:.8f} (≈ 0 ✓)")

    else:
        st.warning("Введите корректные параметры для расчёта.")

    # ── 06 Charts ─────────────────────────────────────────────────────────────
    if not prices.empty:
        st.markdown("")
        st.markdown('<p class="section-header">06 · Исторические графики</p>',
                    unsafe_allow_html=True)

        period_options = {
            "1 месяц":   30,
            "3 месяца":  90,
            "6 месяцев": 180,
            "1 год":     365,
            "2 года":    730,
            "Всё время": None,
        }
        selected_period = st.select_slider(
            "Период отображения",
            options=list(period_options.keys()),
            value="1 год",
        )
        n_days = period_options[selected_period]

        # prices is a guaranteed plain pd.Series after extract_series()
        prices_plot: pd.Series = (
            prices.iloc[-n_days:].copy() if n_days else prices.copy()
        )
        # Strip timezone just in case
        prices_plot.index = pd.to_datetime(prices_plot.index).tz_localize(None)

        tab1, tab2, tab3 = st.tabs([
            "📈 Цена актива",
            "💰 Цены опционов во времени",
            "📊 Чувствительность к цене",
        ])

        # ── Tab 1: historical price ───────────────────────────────────────────
        with tab1:
            st.markdown("**Историческая цена актива**")

            price_df = pd.DataFrame(
                {
                    "Цена актива": prices_plot.values.astype(float),
                    "Страйк K":    float(K),
                },
                index=prices_plot.index,
            )
            st.line_chart(price_df, color=["#e3b341", "#8b949e"], height=340)

            s1, s2, s3, s4 = st.columns(4)
            cur = float(prices_plot.iloc[-1])
            mn  = float(prices_plot.min())
            mx  = float(prices_plot.max())
            chg = (cur / float(prices_plot.iloc[0]) - 1) * 100
            s1.metric("Текущая цена",   f"{cur:.2f}")
            s2.metric("Мин за период",  f"{mn:.2f}")
            s3.metric("Макс за период", f"{mx:.2f}")
            s4.metric("Изм. за период", f"{chg:+.2f}%")

        # ── Tab 2: option price history ───────────────────────────────────────
        with tab2:
            st.markdown("**Расчётная цена колл и пут опционов на каждую дату**")
            st.caption(
                "Для каждой исторической цены считается цена опциона "
                "по модели Блэка-Шоулса с текущими K, r, σ, T."
            )

            if call_price is not None:
                hist_calls, hist_puts = [], []
                for s_val in prices_plot.values:
                    c, p = black_scholes(float(s_val), K, T_val, r, sigma_input)
                    hist_calls.append(c if c is not None else np.nan)
                    hist_puts.append(p if p is not None else np.nan)

                options_df = pd.DataFrame(
                    {"Колл": hist_calls, "Пут": hist_puts},
                    index=prices_plot.index,
                )
                st.line_chart(options_df, color=["#3fb950", "#f85149"], height=340)

                valid_calls = [x for x in hist_calls if not np.isnan(x)]
                valid_puts  = [x for x in hist_puts  if not np.isnan(x)]
                o1, o2, o3, o4 = st.columns(4)
                o1.metric("Колл сейчас", f"{call_price:.4f}")
                o2.metric("Пут сейчас",  f"{put_price:.4f}")
                o3.metric("Колл макс",   f"{max(valid_calls):.4f}" if valid_calls else "—")
                o4.metric("Пут макс",    f"{max(valid_puts):.4f}"  if valid_puts  else "—")
            else:
                st.warning("Задайте корректные параметры для расчёта.")

        # ── Tab 3: price sensitivity ──────────────────────────────────────────
        with tab3:
            st.markdown("**Цена опциона в зависимости от цены актива**")
            st.caption("Сечение при фиксированных T, r, σ. Ось X — цена базового актива.")

            if call_price is not None:
                S_range    = np.linspace(S * 0.4, S * 1.6, 300)
                sens_calls = [black_scholes(s, K, T_val, r, sigma_input)[0] for s in S_range]
                sens_puts  = [black_scholes(s, K, T_val, r, sigma_input)[1] for s in S_range]

                sens_df = pd.DataFrame(
                    {"Колл": sens_calls, "Пут": sens_puts},
                    index=np.round(S_range, 4),
                )
                sens_df.index.name = "Цена актива"
                st.line_chart(sens_df, color=["#3fb950", "#f85149"], height=340)
                st.caption(f"S₀ = {S:.2f}  |  Страйк K = {K:.2f}")
            else:
                st.warning("Задайте корректные параметры для расчёта.")