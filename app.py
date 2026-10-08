import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# 1. IMMERSIVE DARK-NEON THEME
# ==========================================
st.set_page_config(page_title="Customer Intelligence Hub", layout="wide", page_icon="🛰️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Inter:wght@400;600&display=swap');
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {
        background: radial-gradient(1100px 600px at 85% -10%, rgba(0,255,200,0.10), transparent),
                    radial-gradient(900px 500px at 5% 110%, rgba(140,90,255,0.14), transparent),
                    #070b14;
    }
    h1, h2, h3 {font-family: 'Orbitron', sans-serif; color: #e8f0f8;}
    .hero-title {
        font-size: 2.6rem; font-weight: 700; letter-spacing: 5px; text-align: center;
        background: linear-gradient(90deg, #00ffc8, #4d9dff, #b07bff);
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .hero-sub {color: #8b96a5; text-align: center; letter-spacing: 3px; font-size: 0.78rem; text-transform: uppercase;}
    .glass-card {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px; padding: 16px 20px; backdrop-filter: blur(12px);
    }
    .kpi-label {color: #8b96a5; font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase; margin: 0;}
    .kpi-value {font-size: 1.7rem; font-weight: 700; margin: 6px 0 0 0; font-family: 'Orbitron', sans-serif;}
</style>
""", unsafe_allow_html=True)

def kpi_html(label, value, accent):
    return ('<div class="glass-card" style="border-color:' + accent + '44; box-shadow:0 0 22px ' + accent + '22;">'
            '<p class="kpi-label">' + label + '</p>'
            '<p class="kpi-value" style="color:' + accent + ';">' + value + '</p></div>')

def style_fig(fig):
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c7d0dc"),
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ==========================================
# 2. LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("customers_with_clv.csv")
    if "Customer ID" in df.columns:
        df = df.rename(columns={"Customer ID": "CustomerID"})
    return df

df = load_data()

PALETTE = ["#00ffc8", "#4d9dff", "#b07bff", "#ffd166", "#ff7b54", "#8d99ae"]
segments = sorted(df["Segment"].dropna().unique())
color_map = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(segments)}

# ==========================================
# 3. HERO HEADER + KPI GLASS CARDS
# ==========================================
st.markdown('<p class="hero-title">CUSTOMER INTELLIGENCE HUB</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">RFM Clustering • K-Means Segmentation • Predictive Lifetime Value</p>', unsafe_allow_html=True)
st.markdown("")

c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi_html("Customers Tracked", f"{len(df):,}", "#00ffc8"), unsafe_allow_html=True)
c2.markdown(kpi_html("Total Predicted 6M CLV", f"${df['predicted_CLV_6m'].sum():,.0f}", "#4d9dff"), unsafe_allow_html=True)
c3.markdown(kpi_html("Avg CLV / Customer", f"${df['predicted_CLV_6m'].mean():,.2f}", "#b07bff"), unsafe_allow_html=True)
c4.markdown(kpi_html("Active Segments", str(len(segments)), "#ffd166"), unsafe_allow_html=True)

# ==========================================
# 4. INTERACTIVE CONTROL DECK
# ==========================================
st.markdown("")
st.markdown('<p class="hero-sub" style="text-align:left;">⚙️ CONTROL DECK</p>', unsafe_allow_html=True)
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1, 1, 1])

with ctrl1:
    ALL = "🌌 All Segments"
    choice = st.radio("Segment Focus", [ALL] + segments)
with ctrl2:
    x_axis = st.selectbox("Bubble X-Axis", ["Recency", "Frequency"])
with ctrl3:
    y_axis = st.selectbox("Bubble Y-Axis", ["Monetary", "predicted_CLV_6m"])
with ctrl4:
    top_n = st.slider("Top Customers to Show", 5, 50, 15)

scope = df if choice == ALL else df[df["Segment"] == choice]

# ==========================================
# 5. VISUALIZATION GRID
# ==========================================
row1a, row1b = st.columns([2, 1])

with row1a:
    fig_scatter = px.scatter(scope, x=x_axis, y=y_axis, size="Monetary", color="Segment",
                             color_discrete_map=color_map, opacity=0.65,
                             hover_data=["CustomerID", "predicted_CLV_6m"],
                             title="RFM CONSTELLATION MAP (bubble size = Monetary)")
    st.plotly_chart(style_fig(fig_scatter), use_container_width=True)

with row1b:
    counts = df["Segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Count"]
    fig_pie = px.pie(counts, names="Segment", values="Count", hole=0.62,
                     color="Segment", color_discrete_map=color_map, title="SEGMENT MIX")
    fig_pie.update_traces(textinfo="percent", hovertemplate="<b>%{label}</b><br>%{value} customers<extra></extra>")
    st.plotly_chart(style_fig(fig_pie), use_container_width=True)

row2a, row2b = st.columns(2)

with row2a:
    clv_by = df.groupby("Segment")["predicted_CLV_6m"].mean().reset_index()
    fig_bar = px.bar(clv_by, x="Segment", y="predicted_CLV_6m", color="Segment",
                     color_discrete_map=color_map, title="AVG PREDICTED 6-MONTH CLV BY SEGMENT")
    st.plotly_chart(style_fig(fig_bar), use_container_width=True)

with row2b:
    top_c = scope.groupby("Country")["predicted_CLV_6m"].sum().sort_values(ascending=False).head(8).reset_index()
    fig_country = px.bar(top_c, x="predicted_CLV_6m", y="Country", orientation="h",
                         color="predicted_CLV_6m", color_continuous_scale="teal",
                         title="TOP COUNTRIES BY TOTAL CLV (" + ("ALL" if choice == ALL else choice.upper()) + ")")
    st.plotly_chart(style_fig(fig_country), use_container_width=True)

# ==========================================
# 6. CUSTOMER SCANNER + STRATEGY PANEL
# ==========================================
row3a, row3b = st.columns([1, 1])

with row3a:
    st.markdown('<p class="hero-sub" style="text-align:left;">🔎 CUSTOMER SCANNER</p>', unsafe_allow_html=True)
    cid = st.text_input("Enter a Customer ID", placeholder="e.g. 12345")
    if cid.strip():
        match = df[df["CustomerID"].astype(str) == cid.strip()]
        if len(match) > 0:
            r = match.iloc[0]
            pct = (df["predicted_CLV_6m"] < r["predicted_CLV_6m"]).mean() * 100
            card = ('<div class="glass-card" style="border-color:' + color_map.get(r["Segment"], "#fff") + '66;">'
                    '<p class="kpi-value" style="color:' + color_map.get(r["Segment"], "#fff") + ';">' + str(r["Segment"]) + '</p>'
                    '<p class="kpi-label">Customer #' + str(r["CustomerID"]) + ' • ' + str(r["Country"]) + '</p>'
                    '<p style="color:#c7d0dc; margin-top:10px;">Recency: <b>' + f"{r['Recency']:.0f}" + ' days</b> | Frequency: <b>' + f"{r['Frequency']:.0f}" + ' orders</b> | Monetary: <b>$' + f"{r['Monetary']:,.0f}" + '</b></p>'
                    '<p style="color:#00ffc8; font-size:1.2rem; font-weight:700;">Predicted 6M CLV: $' + f"{r['predicted_CLV_6m']:,.2f}" + '</p>'
                    '<p class="kpi-label">Beats ' + f"{pct:.0f}" + '% of all customers in predicted value</p></div>')
            st.markdown(card, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Customer ID not found in database.")

with row3b:
    st.markdown('<p class="hero-sub" style="text-align:left;">💡 AI STRATEGY BRIEFING</p>', unsafe_allow_html=True)
    strategies = {
        "Champions": "Reward loyalty with VIP early-access and referral incentives. They are your brand advocates.",
        "Loyal Customers": "Upsell higher-value categories and loyalty tiers to push them toward Champion status.",
        "New / Potential": "Nurture with onboarding email flows and a second-purchase discount within 30 days.",
        "At Risk (Valuable but Sleeping)": "Win back with personalized 'We miss you' campaigns and time-limited offers.",
        "Hibernating / Lost": "Low-cost reactivation emails only. Exclude from expensive paid campaigns.",
    }
    brief = [choice] if choice != ALL else segments
    for s in brief:
        if s in strategies:
            st.markdown('<div class="glass-card" style="margin-bottom:8px;"><b style="color:' + color_map.get(s, "#fff") + ';">' + s + ':</b> <span style="color:#c7d0dc;">' + strategies[s] + '</span></div>', unsafe_allow_html=True)

# ==========================================
# 7. TOP CUSTOMERS LEADERBOARD
# ==========================================
st.markdown('<p class="hero-sub" style="text-align:left;">🏆 VALUE LEADERBOARD</p>', unsafe_allow_html=True)
top = scope.sort_values("predicted_CLV_6m", ascending=False).head(top_n)
st.dataframe(top[["CustomerID", "Country", "Segment", "Recency", "Frequency", "Monetary", "predicted_CLV_6m"]],
             use_container_width=True, hide_index=True)