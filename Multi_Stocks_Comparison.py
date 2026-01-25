from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ----------------------------
# Parsing / settings helpers
# ----------------------------

def parse_ticker_lines(raw: str) -> list[str]:
    lines = [ln.strip() for ln in (raw or "").splitlines()]
    tickers = [t for t in lines if t]
    # de-dupe, preserve order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def resolve_date_preset(preset: str, start_custom: date, end_custom: date) -> tuple[date, date]:
    today = date.today()
    if preset == "Custom":
        return start_custom, end_custom
    if preset == "Last 3 months":
        return today - timedelta(days=90), today
    if preset == "Last 6 months":
        return today - timedelta(days=180), today
    if preset == "YTD":
        return date(today.year, 1, 1), today
    if preset == "Last 12 months":
        return today - timedelta(days=365), today
    if preset == "Last 24 months":
        return today - timedelta(days=730), today
    if preset == "Last 36 months":
        return today - timedelta(days=1095), today
    return start_custom, end_custom


def fmt_d(d: date) -> str:
    return d.strftime("%d/%m/%Y")


# ----------------------------
# Yahoo metadata + currency
# ----------------------------

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_symbol_meta(symbol: str) -> dict:
    """
    Best-effort metadata (cached): currency + name.
    """
    t = yf.Ticker(symbol)
    meta = {"symbol": symbol, "currency": None, "name": None}

    try:
        meta["currency"] = getattr(t.fastinfo, "currency", None)
    except Exception:
        meta["currency"] = None

    if not meta["currency"]:
        try:
            meta["currency"] = (t.info or {}).get("currency", None)
        except Exception:
            meta["currency"] = None

    try:
        info = t.info or {}
        meta["name"] = info.get("longName") or info.get("shortName")
    except Exception:
        meta["name"] = None

    return meta


def currency_factor_to_major_units(yahoo_currency: str | None) -> tuple[float, str]:
    """
    Convert Yahoo 'GBp'/'GBX' (pence) into pounds-equivalent major units.
    Returns: (multiplier, explanation)
    """
    if yahoo_currency in {"GBp", "GBX"}:
        return 1.0 / 100.0, "Converted from pence (GBp/GBX) to pounds-equivalent by /100"
    if yahoo_currency == "GBP":
        return 1.0, "Already in pounds (GBP)"
    if yahoo_currency:
        return 1.0, f"No conversion applied (Yahoo currency: {yahoo_currency})"
    return 1.0, "Currency unknown (no conversion applied)"


# ----------------------------
# Yahoo download (Close only)
# ----------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_yahoo_close(symbols: list[str], start: date, end: date) -> tuple[pd.DataFrame, list[dict]]:
    """
    Download auto-adjusted Close series for symbols.
    Returns (close_df, issues) where issues is a list of {symbol, problem}.
    """
    if not symbols:
        return pd.DataFrame(), [{"symbol": "", "problem": "No symbols provided"}]

    # yfinance end is often exclusive; add 1 day so UI "end" is effectively inclusive
    end_plus = end + timedelta(days=1)

    data = yf.download(
        symbols,
        start=start,
        end=end_plus,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    issues: list[dict] = []

    if data is None or getattr(data, "empty", True):
        issues.append({"symbol": ",".join(symbols),
                      "problem": "No data returned for any symbol"})
        return pd.DataFrame(), issues

    # MultiIndex for multiple symbols; single-index for one symbol
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            issues.append({"symbol": ",".join(
                symbols), "problem": "Expected Close in yfinance output but not found"})
            return pd.DataFrame(), issues
        close = data["Close"].copy()
    else:
        if "Close" in data.columns:
            close = data["Close"].to_frame()
            close.columns = [symbols[0]]
        elif "Adj Close" in data.columns:
            close = data["Adj Close"].to_frame()
            close.columns = [symbols[0]]
        else:
            issues.append(
                {"symbol": symbols[0], "problem": "Neither Close nor Adj Close found in yfinance output"})
            return pd.DataFrame(), issues

    # Ensure all requested symbols exist as columns (if missing, add all-NaN)
    for s in symbols:
        if s not in close.columns:
            close[s] = np.nan
            issues.append(
                {"symbol": s, "problem": "Symbol missing from Yahoo download (column added as all-NaN)"})

    close = close.sort_index()
    close = close[symbols]  # enforce consistent column order
    return close, issues


# ----------------------------
# Missing-history handling
# ----------------------------

def backfill_leading_flat(close: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    If a symbol has no prices at the requested start but does later,
    fill the leading region with the first valid price so growth is zero
    until real data begins.

    Returns: (filled_close, missing_ranges)
    """
    close = close.sort_index().copy()
    missing_ranges: list[dict] = []

    if close.empty:
        return close, missing_ranges

    first_idx = close.index.min()
    last_idx = close.index.max()

    for s in close.columns:
        ser = close[s]
        if ser.isna().all():
            missing_ranges.append({
                "symbol": s,
                "type": "nodata",
                "start": first_idx,
                "end": last_idx,
            })
            continue

        first_valid = ser.first_valid_index()
        if first_valid is not None and first_valid > first_idx:
            missing_ranges.append({
                "symbol": s,
                "type": "leadingnan",
                "start": first_idx,
                "end": first_valid - pd.Timedelta(days=1),
                "used_price_from": first_valid,
            })
            fillvalue = close.at[first_valid, s]
            close.loc[(close.index >= first_idx) & (
                close.index < first_valid), s] = fillvalue

    # Fill internal gaps too (handles sporadic missing values)
    close = close.ffill()

    return close, missing_ranges


# ----------------------------
# Spike cleaning
# ----------------------------

def clean_daily_spikes_flat(
    close: pd.DataFrame,
    threshold: float = 0.25,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    If abs(1-day move) > threshold, replace today's price with yesterday's price (flat).
    Returns: (clean_close, corrections)
    """
    close = close.sort_index().copy()
    corrections: list[dict] = []

    if close.empty:
        return close, corrections

    for sym in close.columns:
        s = close[sym]
        prev_val = None

        for ts in s.index:
            val = s.at[ts]
            if pd.isna(val):
                continue

            if prev_val is None:
                prev_val = float(val)
                continue

            if prev_val == 0:
                prev_val = float(val)
                continue

            pct = (float(val) / prev_val) - 1.0
            if np.isfinite(pct) and abs(pct) > threshold:
                old = float(val)
                new = float(prev_val)
                s.at[ts] = new
                corrections.append({
                    "symbol": sym,
                    "date": ts,
                    "pct_move": pct,
                    "old_price": old,
                    "new_price": new,
                })
                # keep prev_val unchanged (flat)
            else:
                prev_val = float(val)

        close[sym] = s

    return close, corrections


# ----------------------------
# Transformations for charting
# ----------------------------

def compute_rebased_index(close: pd.DataFrame, base_value: float = 100.0) -> pd.DataFrame:
    out = {}
    for c in close.columns:
        s = close[c].dropna()
        if s.empty:
            continue
        out[c] = (s / s.iloc[0]) * base_value
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_index()


def compute_cum_return(close: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in close.columns:
        s = close[c].dropna()
        if s.empty:
            continue
        out[c] = (s / s.iloc[0]) - 1.0
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_index()


# ----------------------------
# Plot + export
# ----------------------------

def plot_lines(df: pd.DataFrame, title: str, y_label: str, percent: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col in df.columns:
        y = df[col] * 100.0 if percent else df[col]
        ax.plot(df.index, y, label=col, linewidth=1.6)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=2, fontsize=9)
    fig.autofmt_xdate()
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ----------------------------
# Notes generation (new)
# ----------------------------

def build_notes_lines(
    start_date: date,
    end_date: date,
    missing_ranges: list[dict],
    corrections: list[dict],
    spike_threshold_pct: int,
    max_dates_per_symbol: int = 12,
) -> list[str]:
    """
    Returns a list of human-readable note lines to display under the chart.
    """
    lines: list[str] = []

    # Missing/backfill summary
    if not missing_ranges:
        lines.append(
            f"No prices missing for the period {fmt_d(start_date)} to {fmt_d(end_date)}.")
    else:
        any_leading = any(
            m.get("type") == "leadingnan" for m in missing_ranges)
        any_nodata = any(m.get("type") == "nodata" for m in missing_ranges)

        if any_leading:
            for m in missing_ranges:
                if m.get("type") != "leadingnan":
                    continue
                s = m["symbol"]
                start = pd.to_datetime(m["start"]).date()
                end = pd.to_datetime(m["end"]).date()
                used = pd.to_datetime(m["used_price_from"]).date()
                lines.append(
                    f"{s} missing prices from {fmt_d(start)} to {fmt_d(end)}; backfilled using the price from {fmt_d(used)} (assumed zero growth)."
                )

        if any_nodata:
            for m in missing_ranges:
                if m.get("type") != "nodata":
                    continue
                s = m["symbol"]
                start = pd.to_datetime(m["start"]).date()
                end = pd.to_datetime(m["end"]).date()
                lines.append(
                    f"{s} has no usable Yahoo price history between {fmt_d(start)} and {fmt_d(end)}.")

    # Spike-clean summary
    if not corrections:
        lines.append(
            f"No spike flattening was applied (threshold {spike_threshold_pct}% daily move).")
    else:
        lines.append(
            f"Possible data-quality spikes were flattened when the 1-day move exceeded {spike_threshold_pct}% (today's price replaced with yesterday's)."
        )

        by_sym: dict[str, list[date]] = {}
        for c in corrections:
            sym = c["symbol"]
            d = pd.to_datetime(c["date"]).date()
            by_sym.setdefault(sym, []).append(d)

        for sym in sorted(by_sym.keys()):
            dts = sorted(set(by_sym[sym]))
            shown = dts[:max_dates_per_symbol]
            dates_str = ", ".join(dt.strftime("%d/%m/%Y") for dt in shown)
            more = "" if len(
                dts) <= max_dates_per_symbol else f" (+{len(dts) - max_dates_per_symbol} more)"
            lines.append(f"{sym} flattened on: {dates_str}{more}.")

    return lines


# ----------------------------
# Streamlit App
# ----------------------------

st.set_page_config(page_title="Yahoo! Charts", layout="wide")
st.title("Yahoo! Charts")

with st.sidebar:
    st.header("Tickers")

    raw_tickers = st.text_area(
        "Enter tickers (one per line)",
        value="VHYL.L\nCSP1.L\nSGLN.L\nFGQI.L\nJEGI.L",
        height=160,
        help="Example: VHYL.L, CSP1.L, SGLN.L, FGQI.L, JEGI.L",
    )

    st.header("Settings")

    benchmark = st.text_input("Benchmark ticker", value="^GSPC")

    date_preset = st.selectbox(
        "Date range",
        ["Custom", "Last 3 months", "Last 6 months", "YTD",
            "Last 12 months", "Last 24 months", "Last 36 months"],
        index=4,
    )

    today = date.today()
    default_start = today - timedelta(days=365)

    start_custom = st.date_input("Start date", value=default_start)
    end_custom = st.date_input("End date", value=today)

    start_date, end_date = resolve_date_preset(
        date_preset, start_custom, end_custom)

    chart_mode = st.selectbox(
        "Chart mode",
        ["Cumulative return (%)", "Rebased index (start=100)"],
        index=0,
    )

    st.subheader("Yahoo spike cleaning")
    enable_spike_clean = st.checkbox(
        "Flatten suspicious 1-day spikes",
        value=True,
        help="If abs(1-day move) exceeds the threshold, replace that day's price with the prior day's price.",
    )
    spike_threshold_pct = st.slider(
        "Spike threshold (%)",
        min_value=5,
        max_value=80,
        value=25,
        step=5,
    )

    show_currency_table = st.checkbox(
        "Show currency / pence-pound handling", value=False)

    run = st.button("Update chart", type="primary")


if not run:
    st.info("Enter tickers on the left, adjust settings, then click “Update chart”.")
    st.stop()

tickers = parse_ticker_lines(raw_tickers)
if not tickers:
    st.error("Please enter at least one ticker.")
    st.stop()

benchmark = benchmark.strip()
if not benchmark:
    st.error("Please enter a benchmark ticker.")
    st.stop()

if end_date <= start_date:
    st.error("End date must be after start date.")
    st.stop()

symbols = tickers + [benchmark]

with st.spinner("Downloading prices from Yahoo..."):
    close_raw, issues = fetch_yahoo_close(symbols, start_date, end_date)

if close_raw.empty:
    st.error("No price data returned.")
    st.stop()

# Currency table + pence->pounds conversion where appropriate
meta_rows = []
factors = {}
for s in symbols:
    meta = get_symbol_meta(s)
    factor, note = currency_factor_to_major_units(meta.get("currency"))
    factors[s] = factor
    meta_rows.append({
        "Symbol": s,
        "Name": meta.get("name") or "",
        "Yahoo currency": meta.get("currency") or "",
        "Applied factor": factor,
        "Note": note,
    })

meta_df = pd.DataFrame(meta_rows)

close = close_raw.copy()
for s, f in factors.items():
    if s in close.columns and f != 1.0:
        close[s] = close[s] * f

# Warn about download problems
if issues:
    st.warning("Some Yahoo download issues occurred:")
    for it in issues:
        st.write(
            f"- {it.get('symbol', '?')}: {it.get('problem', 'Unknown issue')}")

# Backfill leading gaps: assume flat price until first print
close_filled, missing_ranges = backfill_leading_flat(close)

# Drop symbols that are truly all-NaN
close_filled = close_filled.dropna(axis=1, how="all")

if benchmark not in close_filled.columns:
    st.error(f"Benchmark {benchmark} has no usable data in this period.")
    st.stop()

available_tickers = [t for t in tickers if t in close_filled.columns]
if not available_tickers:
    st.error("None of the entered tickers have usable data in this period.")
    st.stop()

# Keep benchmark + tickers in desired order
ordered_cols = available_tickers + [benchmark]
close_filled = close_filled[ordered_cols]

# Spike cleaning
corrections: list[dict] = []
if enable_spike_clean:
    close_filled, corrections = clean_daily_spikes_flat(
        close_filled,
        threshold=float(spike_threshold_pct) / 100.0,
    )

if show_currency_table:
    with st.expander("Currency + pence/pounds handling", expanded=False):
        st.dataframe(meta_df, use_container_width=True)

# Compute series for plotting
if chart_mode == "Cumulative return (%)":
    plot_df = compute_cum_return(close_filled)
    title = f"Cumulative return (rebased to 0) — {start_date} to {end_date}"
    ylab = "Cumulative return (%)"
    percent = True
else:
    plot_df = compute_rebased_index(close_filled, base_value=100.0)
    title = f"Rebased index (start=100) — {start_date} to {end_date}"
    ylab = "Index level"
    percent = False

if plot_df.empty or plot_df.shape[1] < 2:
    st.error("Not enough data to plot (need at least one ticker plus the benchmark).")
    st.stop()

# Rename benchmark label for clarity
plot_df = plot_df.rename(columns={benchmark: f"Benchmark: {benchmark}"})

fig = plot_lines(plot_df, title=title, y_label=ylab, percent=percent)
st.pyplot(fig, use_container_width=True)

png_bytes = fig_to_png_bytes(fig)
st.download_button(
    "Download chart as PNG",
    data=png_bytes,
    file_name=f"chart_{start_date}_{end_date}.png",
    mime="image/png",
)

# Notes below chart (new)
st.subheader("Notes / data quality")
notes = build_notes_lines(
    start_date=start_date,
    end_date=end_date,
    missing_ranges=missing_ranges,
    corrections=corrections,
    spike_threshold_pct=spike_threshold_pct,
    max_dates_per_symbol=12,
)
for line in notes:
    st.write(f"- {line}")

