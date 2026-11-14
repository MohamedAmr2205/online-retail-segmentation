"""
Retail Pulse | Know your customers. Move your products.
Online Retail capstone dashboard.  Author: Mohamed Amr Mohamed

Run from the repo root:   streamlit run app/streamlit_app.py
Reads only data/processed/*.csv, which the notebook writes (Appendix: Export processed tables).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Retail Pulse", page_icon="📊", layout="wide")

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# name -> (file, required columns)
FILES = {
    "customers": ("customers_segmented.csv",
                  ["CustomerID", "Recency", "Frequency", "Monetary", "R", "F", "M", "Segment", "FinalSegment", "country"]),
    "profile": ("segment_profile.csv", ["FinalSegment", "customers", "revenue", "customers_%", "revenue_%"]),
    "matrix": ("business_matrix.csv", ["segment", "evidence", "strategy", "kpi_to_test"]),
    "monthly": ("monthly_revenue.csv", ["YearMonth", "revenue", "orders"]),
    "country": ("country_summary.csv", ["Country", "revenue", "orders", "customers"]),
    "products": ("top_products.csv", ["StockCode", "name", "units", "revenue", "n_invoices"]),
    "heat": ("weekday_hour.csv", ["DayOfWeek"]),
    "kpis": ("kpis.csv", ["net_revenue", "orders", "identified_customers"]),
    "insights": ("insights.csv", ["insight"]),
    "log": ("decision_log.csv", ["step", "decision", "rows_affected", "business_impact"]),
}

FINAL_ORDER = ["Whales", "Core Loyal", "Due-Now Regulars", "Recent Light Buyers", "Dormant One-Timers"]
RULE_ORDER = ["Champions", "Loyal", "New Inflow", "About to Sleep", "At-Risk High-Value", "Lost / Hibernating"]
SEG_COLORS = {
    "Whales": "#FF2E63", "Core Loyal": "#08D9D6", "Due-Now Regulars": "#F9C74F",
    "Recent Light Buyers": "#7BD389", "Dormant One-Timers": "#A78BFA",
    "Champions": "#FF2E63", "Loyal": "#08D9D6", "New Inflow": "#7BD389",
    "About to Sleep": "#F9C74F", "At-Risk High-Value": "#F3722C", "Lost / Hibernating": "#A78BFA",
}
ACCENT, ACCENT2 = "#08D9D6", "#FF2E63"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
COUNTRY_FIX = {"EIRE": "Ireland", "RSA": "South Africa", "USA": "United States", "Channel Islands": "Jersey",
               "Unspecified": None, "European Community": None}

THEMES = {
    "Midnight": dict(bg="#071426", card="#0E2340", text="#EAF2FF", muted="#9DB3D1", border="#1B3A63", plotly="plotly_dark"),
    "Slate":    dict(bg="#8E9BB8", card="#E9EEF8", text="#14213D", muted="#4A5878", border="#C3CCE0", plotly="plotly_white"),
    "Daylight": dict(bg="#F3F6FC", card="#FFFFFF", text="#14213D", muted="#5B6B8C", border="#DDE4F2", plotly="plotly_white"),
}

# ----------------------------------------------------------------------------
# Sidebar + theme
# ----------------------------------------------------------------------------
PAGES = ["Overview", "Product Analysis", "Customer Segments", "Customer Lookup", "Insights & Actions"]
with st.sidebar:
    st.markdown("### 📊 Retail Pulse")
    st.caption("Know your customers. Move your products.")
    theme_name = st.radio("Theme", list(THEMES), index=0, horizontal=True)
    page = st.radio("Go to", PAGES)
    st.divider()
    st.caption("Built by **Mohamed Amr Mohamed**")

T = THEMES[theme_name]

st.markdown(f"""
<style>
.stApp {{ background:{T['bg']}; color:{T['text']}; }}
header[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stSidebar"] {{ background:{T['card']}; border-right:1px solid {T['border']}; }}
.stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp span,
.stApp li, .stApp [data-testid="stCaptionContainer"], [data-testid="stSidebar"] * {{ color:{T['text']}; }}
.block-container {{ padding-top:1.6rem; max-width:1400px; }}
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"],
[data-testid="stTextInput"] > div > div, [data-baseweb="select"] > div, [data-baseweb="input"] > div {{
    background:{T['card']} !important; border-color:{T['border']} !important; }}
[data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input, [data-testid="stTextInput"] input,
[data-testid="stSelectbox"] [role="group"] *, [data-baseweb="select"] * {{ color:{T['text']} !important; }}
[role="listbox"], [role="listbox"] [role="option"], [data-testid="stSelectboxDropdown"],
[data-testid="stMultiSelectDropdown"], div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li {{
    background:{T['card']} !important; color:{T['text']} !important; }}
input::placeholder {{ color:{T['muted']} !important; opacity:1; }}
[role="option"][data-focused], [role="option"]:hover {{ background:{T['border']} !important; }}
[data-testid="stMultiSelect"] [data-tag] {{ background:{ACCENT2} !important; color:#fff !important; }}
[data-testid="stMultiSelect"] [data-tag] * {{ color:#fff !important; }}
.card {{ background:{T['card']}; border:1px solid {T['border']}; border-radius:16px; padding:16px 18px; height:100%; }}
.kpi-label {{ font-size:.75rem; letter-spacing:.07em; text-transform:uppercase; color:{T['muted']} !important; }}
.kpi-value {{ font-size:1.85rem; font-weight:700; line-height:1.15; }}
.kpi-sub {{ font-size:.8rem; color:{T['muted']} !important; margin-top:2px; }}
.hero {{ border-radius:18px; padding:22px 26px; margin-bottom:16px; border:1px solid {T['border']};
         background:linear-gradient(120deg, {ACCENT}33, {ACCENT2}26); }}
.hero h1 {{ margin:0; font-size:1.8rem; line-height:1.2; }}
.hero p {{ margin:6px 0 0 0; color:{T['muted']} !important; font-size:1rem; }}
.pill {{ display:inline-block; padding:3px 12px; border-radius:999px; font-weight:600; font-size:.8rem;
         color:#fff !important; margin-right:6px; }}
.section-gap {{ margin-top:10px; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(
        f'<div class="card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(template=T["plotly"], height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=T["text"]),
                      margin=dict(l=10, r=10, t=55, b=10), legend_title_text="")
    fig.update_xaxes(gridcolor=T["border"])
    fig.update_yaxes(gridcolor=T["border"])
    return fig


def show(fig: go.Figure, height: int = 380, target=None) -> None:
    (target or st).plotly_chart(style(fig, height), width="stretch", theme=None)


def pill(seg: str) -> str:
    return f'<span class="pill" style="background:{SEG_COLORS.get(seg, "#6b7a99")}">{seg}</span>'


def money(x: float) -> str:
    return f"£{x/1e6:,.2f}M" if abs(x) >= 1e6 else f"£{x:,.0f}"


@st.cache_data(show_spinner="Loading data...")
def load_csv(fname: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / fname)


def load_all():
    data, problems = {}, []
    for key, (fname, cols) in FILES.items():
        if not (DATA_DIR / fname).exists():
            problems.append(f"Missing file: `data/processed/{fname}`")
            continue
        df = load_csv(fname)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            problems.append(f"`{fname}` is missing columns: {missing}")
        data[key] = df
    return data, problems


data, problems = load_all()
if problems:
    st.error("The dashboard could not load the processed data.")
    for p in problems:
        st.write("•", p)
    st.info("Run the notebook top to bottom (Restart & Run All). The last cell writes these files to `data/processed/`.")
    st.stop()

cust = data["customers"].copy()
cust["Segment"] = pd.Categorical(cust["Segment"], categories=RULE_ORDER, ordered=True)
cust["FinalSegment"] = pd.Categorical(cust["FinalSegment"], categories=FINAL_ORDER, ordered=True)
monthly, country, products = data["monthly"], data["country"], data["products"].copy()
profile, matrix, kp = data["profile"], data["matrix"], data["kpis"].iloc[0]
products["avg_price"] = (products["revenue"] / products["units"].where(products["units"] > 0)).round(2)

# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
if page == "Overview":
    hero("Retail Pulse",
         'Know your customers. Move your products. &nbsp;<span dir="rtl">اعرف عميلك.. وحرّك منتجك</span>')

    c = st.columns(5)
    kpi(c[0], "Net revenue", money(kp["net_revenue"]), f"{kp.get('first_date', '')} to {kp.get('last_date', '')}")
    kpi(c[1], "Orders", f"{kp['orders']:,.0f}", "valid purchase invoices")
    kpi(c[2], "Identified customers", f"{kp['identified_customers']:,.0f}", "with a CustomerID")
    if "return_rate_pct" in kp:
        kpi(c[3], "Return rate", f"{kp['return_rate_pct']:.1f}%", "of gross sales")
    if "uk_share_pct" in kp:
        kpi(c[4], "UK share", f"{kp['uk_share_pct']:.0f}%", "of net product revenue")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    left, right = st.columns([3, 2])
    m = monthly.copy()
    if "partial_month" not in m:
        m["partial_month"] = False
    m["Month"] = np.where(m["partial_month"], "Incomplete month", "Full month")
    fig = px.bar(m, x="YearMonth", y="revenue", color="Month", hover_data=["orders"],
                 color_discrete_map={"Full month": ACCENT, "Incomplete month": ACCENT2},
                 title="Monthly net revenue")
    fig.update_xaxes(type="category", title="")
    fig.update_yaxes(title="")
    show(fig, 380, left)

    cm = country.copy()
    cm["iso_name"] = cm["Country"].map(lambda x: COUNTRY_FIX.get(x, x))
    cm = cm.dropna(subset=["iso_name"])
    hide_uk = right.toggle("Hide the UK on the map (it dominates)", value=True)
    if hide_uk:
        cm = cm[cm["Country"] != "United Kingdom"]
    fig = px.choropleth(cm, locations="iso_name", locationmode="country names", color="revenue",
                        hover_name="Country", hover_data=["orders", "customers"],
                        color_continuous_scale=["#E3E8F2", ACCENT, ACCENT2], title="Net revenue by country")
    fig.update_geos(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=False)
    show(fig, 340, right)

    h = data["heat"].copy()
    hm = h.melt("DayOfWeek", var_name="Hour", value_name="revenue")
    hm["Hour"] = hm["Hour"].astype(int)
    pivot = hm.pivot_table(index="DayOfWeek", columns="Hour", values="revenue", aggfunc="sum")
    pivot = pivot.reindex([d for d in DAYS if d in pivot.index])
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale=["#E3E8F2", ACCENT, ACCENT2],
                    labels=dict(x="Hour of day", y="", color="Revenue"),
                    title="When do customers buy? Revenue by weekday and hour")
    show(fig, 330)
    st.caption("Purchases only. No transactions are recorded on Saturdays in this dataset.")

elif page == "Product Analysis":
    hero("Product Analysis",
         'What sells, what earns, and how concentrated the range is. &nbsp;<span dir="rtl">إيه اللي بيبيع وإيه اللي بيكسّب</span>')
    st.caption("Showing the top 300 products by net revenue (the notebook exports these).")

    f1, f2 = st.columns([1, 2])
    metric = f1.radio("Rank by", ["revenue", "units"], horizontal=True)
    top_n = f2.slider("How many products", 5, 30, 10)

    left, right = st.columns(2)
    top = products.nlargest(top_n, metric).iloc[::-1]
    fig = px.bar(top, x=metric, y="name", orientation="h", hover_data=["units", "revenue", "n_invoices"],
                 title=f"Top {top_n} products by {metric}",
                 color_discrete_sequence=[ACCENT if metric == "revenue" else ACCENT2])
    fig.update_yaxes(title="")
    fig.update_xaxes(title="")
    show(fig, 460, left)

    ranked = products.sort_values("revenue", ascending=False).reset_index(drop=True)
    share = ranked["revenue"].cumsum() / kp["net_revenue"] * 100
    fig = go.Figure(go.Scatter(x=np.arange(1, len(ranked) + 1), y=share, mode="lines",
                               line=dict(color=ACCENT2, width=3), fill="tozeroy",
                               fillcolor="rgba(255,46,99,0.12)"))
    fig.update_layout(title=f"Top {len(ranked)} products = {share.iloc[-1]:.0f}% of net revenue",
                      xaxis_title="Products ranked by revenue", yaxis_title="% of total net revenue")
    show(fig, 460, right)

    sc = products[products["units"] > 0]
    fig = px.scatter(sc, x="avg_price", y="units", size="revenue", hover_name="name", log_x=True, log_y=True,
                     color_discrete_sequence=[ACCENT], size_max=40,
                     title="Price vs volume (bubble = revenue): cheap items sell in bulk, pricey items earn per unit")
    fig.update_xaxes(title="Average price per unit (£, log)")
    fig.update_yaxes(title="Units sold (log)")
    show(fig, 420)

    q = st.text_input("Search a product")
    view = products[products["name"].astype(str).str.contains(q, case=False, na=False)] if q else products
    st.dataframe(view.sort_values("revenue", ascending=False), width="stretch", hide_index=True)

elif page == "Customer Segments":
    hero("Customer Segments",
         'Who they are, what they are worth, and what to do about it. &nbsp;<span dir="rtl">مين العملاء وقيمتهم</span>')

    view_name = st.radio("Segmentation", ["Final segments (Whales + K-Means)", "Rule-based RFM"], horizontal=True)
    seg_col, order = ("FinalSegment", FINAL_ORDER) if view_name.startswith("Final") else ("Segment", RULE_ORDER)

    f1, f2 = st.columns(2)
    segs = f1.multiselect("Segment", order, default=order)
    ctry = f2.multiselect("Country (empty = all)", sorted(cust["country"].dropna().unique()))

    v = cust[cust[seg_col].isin(segs)]
    if ctry:
        v = v[v["country"].isin(ctry)]
    if v.empty:
        st.warning("No customers match these filters.")
        st.stop()

    c = st.columns(4)
    kpi(c[0], "Customers", f"{len(v):,}")
    kpi(c[1], "Identified spend", money(v["Monetary"].sum()))
    kpi(c[2], "Median orders", f"{v['Frequency'].median():.0f}")
    kpi(c[3], "Median recency", f"{v['Recency'].median():.0f} days")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    left, right = st.columns([3, 2])
    smp = v.sample(min(len(v), 3000), random_state=42)
    fig = px.scatter(smp, x="Recency", y="Monetary", color=seg_col, log_y=True, opacity=0.75,
                     color_discrete_map=SEG_COLORS, category_orders={seg_col: order},
                     size="Frequency", size_max=22, hover_data=["CustomerID", "Frequency", "country"],
                     title="Recency vs spend (bubble = orders, spend on log scale)")
    fig.update_xaxes(title="Days since last order")
    fig.update_yaxes(title="Total spend (£, log)")
    show(fig, 460, left)

    prof = v.groupby(seg_col, observed=True).agg(customers=("CustomerID", "size"), spend=("Monetary", "sum")).reset_index()
    prof["% of customers"] = prof["customers"] / prof["customers"].sum() * 100
    prof["% of spend"] = prof["spend"] / prof["spend"].sum() * 100
    fig = px.bar(prof.melt(seg_col, ["% of customers", "% of spend"]), x=seg_col, y="value", color="variable",
                 barmode="group", color_discrete_sequence=[ACCENT, ACCENT2], title="Share of customers vs share of spend")
    fig.update_xaxes(title="")
    fig.update_yaxes(title="%")
    show(fig, 460, right)

    st.markdown("#### Segment profile")
    table = v.groupby(seg_col, observed=True).agg(
        customers=("CustomerID", "size"), median_recency_days=("Recency", "median"),
        median_orders=("Frequency", "median"), median_spend=("Monetary", "median"),
        total_spend=("Monetary", "sum")).round(0)
    table["spend_per_customer"] = (table["total_spend"] / table["customers"]).round(0)
    st.dataframe(table, width="stretch")

    with st.expander("How do the two segmentations relate?"):
        ct = pd.crosstab(cust["Segment"], cust["FinalSegment"])
        fig = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale=["#E3E8F2", ACCENT, ACCENT2],
                        labels=dict(x="Final segment", y="Rule-based segment", color="Customers"),
                        title="Rule-based RFM (rows) vs final segments (columns)")
        show(fig, 420)
        st.caption("Example: rule-based 'Champions' mixes whales with ordinary loyal buyers. The final segmentation separates them.")

    st.download_button("Download filtered customers (CSV)", v.to_csv(index=False).encode(),
                       "customers_filtered.csv", "text/csv")

elif page == "Customer Lookup":
    hero("Customer Lookup",
         'Pick a customer ID and get the full picture in one card. &nbsp;<span dir="rtl">دوّر على أي عميل</span>')

    ids = cust["CustomerID"].astype(str)
    cid = st.selectbox("Customer ID", sorted(ids.tolist()))
    row = cust[ids == cid].iloc[0]
    final, rule = str(row["FinalSegment"]), str(row["Segment"])

    st.markdown(f"#### Customer {cid} &nbsp; {pill(final)} {pill(rule)}", unsafe_allow_html=True)
    st.caption("First pill = final segment, second pill = rule-based RFM segment.")

    c = st.columns(5)
    kpi(c[0], "Last order", f"{row['Recency']:.0f} days ago", f"R score {row['R']}")
    kpi(c[1], "Orders", f"{row['Frequency']:.0f}", f"F score {row['F']}")
    kpi(c[2], "Total spend", money(row["Monetary"]), f"M score {row['M']}")
    kpi(c[3], "Spend rank", f"top {100 - (cust['Monetary'] < row['Monetary']).mean() * 100:.0f}%", "of all customers")
    kpi(c[4], "Country", str(row["country"]))

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    left, right = st.columns(2)

    m = matrix[matrix["segment"] == final]
    if not m.empty:
        a = m.iloc[0]
        left.markdown(
            f'<div class="card"><div class="kpi-label">Recommended action</div>'
            f'<div style="font-size:1.1rem;font-weight:600;margin:6px 0">{a["strategy"]}</div>'
            f'<div class="kpi-sub">Segment evidence: {a["evidence"]}</div>'
            f'<div class="kpi-sub">KPI to test: {a["kpi_to_test"]}</div></div>', unsafe_allow_html=True)

    extra = []
    if "median_gap_days" in row and pd.notna(row["median_gap_days"]):
        gap = row["median_gap_days"]
        state = "past their own usual gap" if row["Recency"] > gap else "within their own usual gap"
        extra.append(f"Usually reorders every {gap:.0f} days; last order {row['Recency']:.0f} days ago ({state}).")
    elif row["Frequency"] == 1:
        extra.append("One-time buyer so far: no reorder pattern to compare with.")
    if "avg_order_value" in row and pd.notna(row["avg_order_value"]):
        extra.append(f"Average order value {money(row['avg_order_value'])}.")
    if "return_ratio" in row and pd.notna(row["return_ratio"]):
        extra.append(f"Returned {row['return_ratio'] * 100:.1f}% of what they bought.")
    left.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    left.markdown(
        '<div class="card"><div class="kpi-label">Behaviour</div>'
        + "".join(f'<div class="kpi-sub" style="font-size:.95rem;margin-top:6px">{t}</div>' for t in extra)
        + "</div>", unsafe_allow_html=True)

    peers = cust[cust["FinalSegment"] == row["FinalSegment"]][["Recency", "Frequency", "Monetary"]].median()
    vals = row[["Recency", "Frequency", "Monetary"]].astype(float)
    comp = pd.DataFrame({
        "metric": ["Recency (days)", "Orders", "Spend (£)"] * 2,
        "who": ["This customer"] * 3 + ["Segment median"] * 3,
        "value": list(vals) + list(peers),
    })
    comp["scaled"] = comp.groupby("metric")["value"].transform(lambda s: s / s.max() * 100 if s.max() else 0)
    fig = px.bar(comp, x="metric", y="scaled", color="who", barmode="group", hover_data=["value"],
                 color_discrete_sequence=[ACCENT2, ACCENT], title="This customer vs their segment median (scaled, bigger = higher)")
    fig.update_xaxes(title="")
    fig.update_yaxes(title="", showticklabels=False)
    show(fig, 340, right)

elif page == "Insights & Actions":
    hero("Insights & Actions",
         'The data story, then what to do with each segment. &nbsp;<span dir="rtl">الحكاية والإجراءات</span>')
    st.caption("Descriptive analysis of one year of data. These are hypotheses to test, not causal claims.")

    st.markdown("#### Segment → strategy matrix")
    st.dataframe(matrix.rename(columns={"segment": "Segment", "evidence": "Evidence", "strategy": "Action",
                                        "kpi_to_test": "KPI to test"}), width="stretch", hide_index=True)

    st.markdown("#### Insights")
    cols = st.columns(2)
    for i, text in enumerate(data["insights"]["insight"].astype(str).tolist()):
        label, _, body = text.partition(":")
        if not body or len(label) > 40:
            label, body = "", text
        cols[i % 2].markdown(
            f'<div class="card" style="margin-bottom:12px"><div class="kpi-label">{i + 1:02d}'
            f'{" · " + label if label else ""}</div><div style="margin-top:4px">{body.strip()}</div></div>',
            unsafe_allow_html=True)

    with st.expander("Data-cleaning decision log"):
        st.dataframe(data["log"], width="stretch", hide_index=True)

st.divider()
st.caption("Retail Pulse · Mohamed Amr Mohamed")
