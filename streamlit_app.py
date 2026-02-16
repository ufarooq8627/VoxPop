"""
VoxPop: AI-Driven Global Brand Sentiment & Crisis Intelligence
Streamlit Dashboard — Task 5: Deployment & Statistics

This dashboard provides an interactive visualization of sentiment analysis
results, hypothesis testing, crisis reports, and entity intelligence.
"""

import matplotlib
matplotlib.use("Agg")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sqlalchemy import create_engine, text
import warnings
import os

warnings.filterwarnings("ignore")

# CONFIG & DB CONNECTION

DB_URL = os.environ.get(
    "VOXPOP_DB_URL",
    "postgresql+psycopg://postgres:YOUR_PASSWORD@127.0.0.1:5432/voxpop"
)

st.set_page_config(
    page_title="VoxPop — Brand Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_engine():
    """Create and cache the SQLAlchemy engine."""
    return create_engine(DB_URL)


@st.cache_data(ttl=600)
def load_reviews(sample_size: int) -> pd.DataFrame:
    """Load reviews from the fact_reviews table with optional sampling.

    Args:
        sample_size: Number of rows to load. Use 0 for all rows.

    Returns:
        DataFrame with review data.
    """
    engine = get_engine()

    if sample_size > 0:
        query = text("""
            SELECT review_id, clean_text, sentiment, new_date, user_id
            FROM fact_reviews
            TABLESAMPLE SYSTEM_ROWS(:n)
        """)
        # Fallback if tsm_system_rows is not available
        try:
            df = pd.read_sql(query, engine, params={"n": sample_size})
        except Exception:
            query = text("""
                SELECT review_id, clean_text, sentiment, new_date, user_id
                FROM fact_reviews
                ORDER BY RANDOM()
                LIMIT :n
            """)
            df = pd.read_sql(query, engine, params={"n": sample_size})
    else:
        query = text("SELECT review_id, clean_text, sentiment, new_date, user_id FROM fact_reviews")
        df = pd.read_sql(query, engine)

    df["new_date"] = pd.to_datetime(df["new_date"], errors="coerce")
    df["target"] = df["sentiment"].apply(lambda x: 1 if x == 4 else 0)
    df["label"] = df["target"].map({1: "Positive", 0: "Negative"})
    return df


@st.cache_data(ttl=600)
def load_crisis_reports() -> pd.DataFrame:
    """Load crisis reports from the crisis_reports table."""
    engine = get_engine()
    try:
        return pd.read_sql(text("SELECT * FROM crisis_reports ORDER BY timestamp DESC"), engine)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_entities() -> pd.DataFrame:
    """Load NER entities from the entity_store table."""
    engine = get_engine()
    try:
        query = text("""
            SELECT entity, label, COUNT(*) AS mentions
            FROM entity_store
            GROUP BY entity, label
            ORDER BY mentions DESC
            LIMIT 20
        """)
        return pd.read_sql(query, engine)
    except Exception:
        return pd.DataFrame()


# CUSTOM STYLING

st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    .main-header p {
        color: #a8b2d1;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .kpi-label {
        color: #a8b2d1;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    .positive { color: #00d2ff; }
    .negative { color: #e94560; }
    .neutral  { color: #f5a623; }
    .total    { color: #a8e6cf; }

    /* Section headers */
    .section-header {
        border-left: 4px solid #e94560;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }

    /* Stat result box */
    .stat-box {
        background: #16213e;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #0f3460;
    }
</style>
""", unsafe_allow_html=True)


# SIDEBAR

with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/combo-chart.png", width=80)
    st.title("⚙️ Controls")
    st.markdown("---")

    sample_size = st.slider(
        "📦 Sample Size",
        min_value=10_000,
        max_value=500_000,
        value=100_000,
        step=10_000,
        help="Number of reviews to load. Larger = slower but more accurate."
    )

    st.markdown("---")
    st.caption("VoxPop v1.0 | Task 5 Dashboard")


# LOAD DATA

try:
    df = load_reviews(sample_size)
except Exception as e:
    st.error(f"❌ Failed to connect to PostgreSQL: {e}")
    st.info("Make sure PostgreSQL is running on `127.0.0.1:5432` with database `voxpop`.")
    st.stop()


# HEADER

st.markdown("""
<div class="main-header">
    <h1>📊 VoxPop Intelligence Dashboard</h1>
    <p>AI-Driven Global Brand Sentiment & Crisis Intelligence</p>
</div>
""", unsafe_allow_html=True)


# 1. KPI METRICS

total = len(df)
pos_count = int((df["target"] == 1).sum())
neg_count = int((df["target"] == 0).sum())
pos_pct = (pos_count / total) * 100 if total > 0 else 0
neg_pct = (neg_count / total) * 100 if total > 0 else 0
nss = pos_pct - neg_pct  # Net Sentiment Score

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-value total">{total:,}</p>
        <p class="kpi-label">Total Reviews Loaded</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-value positive">{pos_count:,} ({pos_pct:.1f}%)</p>
        <p class="kpi-label">✅ Positive Reviews</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-value negative">{neg_count:,} ({neg_pct:.1f}%)</p>
        <p class="kpi-label">🚨 Negative Reviews</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    nss_color = "positive" if nss > 0 else "negative"
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-value {nss_color}">{nss:+.1f}</p>
        <p class="kpi-label">📈 Net Sentiment Score (NSS)</p>
    </div>
    """, unsafe_allow_html=True)


# 2. SENTIMENT DISTRIBUTION — DONUT CHART

st.markdown('<div class="section-header"><h2>🍩 Sentiment Distribution</h2></div>', unsafe_allow_html=True)

col_donut, col_bar = st.columns(2)

with col_donut:
    sentiment_counts = df["label"].value_counts()
    fig_donut = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        hole=0.55,
        color=sentiment_counts.index,
        color_discrete_map={"Positive": "#00d2ff", "Negative": "#e94560"},
    )
    fig_donut.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14),
        showlegend=True,
        height=400,
    )
    fig_donut.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont_size=14,
    )
    st.plotly_chart(fig_donut, use_container_width=True)


# 3. SENTIMENT OVER TIME — STACKED BAR

with col_bar:
    df_time = df.dropna(subset=["new_date"]).copy()
    df_time["week"] = df_time["new_date"].dt.to_period("W").astype(str)

    time_sentiment = df_time.groupby(["week", "label"]).size().reset_index(name="count")

    fig_time = px.bar(
        time_sentiment,
        x="week",
        y="count",
        color="label",
        color_discrete_map={"Positive": "#00d2ff", "Negative": "#e94560"},
        barmode="stack",
        labels={"count": "Number of Reviews", "week": "Week"},
    )
    fig_time.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-45,
        height=400,
        legend_title_text="Sentiment",
    )
    st.plotly_chart(fig_time, use_container_width=True)


# 4. WORD CLOUDS

st.markdown('<div class="section-header"><h2>☁️ Word Clouds</h2></div>', unsafe_allow_html=True)

col_wc_pos, col_wc_neg = st.columns(2)

pos_reviews = df[df["target"] == 1]["clean_text"].dropna()
neg_reviews = df[df["target"] == 0]["clean_text"].dropna()

pos_text = " ".join(pos_reviews.sample(min(20000, len(pos_reviews)), random_state=42)) if len(pos_reviews) > 0 else ""
neg_text = " ".join(neg_reviews.sample(min(20000, len(neg_reviews)), random_state=42)) if len(neg_reviews) > 0 else ""

with col_wc_pos:
    st.subheader("✅ Positive Reviews")
    if pos_text.strip():
        try:
            wc_pos = WordCloud(
                width=800, height=400,
                background_color="#0e1117",
                colormap="cool",
                max_words=150,
            ).generate(pos_text)

            fig_wc_pos, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc_pos.to_array(), interpolation="bilinear")
            ax.axis("off")
            fig_wc_pos.patch.set_facecolor("#0e1117")
            st.pyplot(fig_wc_pos)
            plt.close(fig_wc_pos)
        except Exception as e:
            st.warning(f"Could not generate word cloud: {e}")
    else:
        st.info("No positive review text available.")

with col_wc_neg:
    st.subheader("🚨 Negative Reviews")
    if neg_text.strip():
        try:
            wc_neg = WordCloud(
                width=800, height=400,
                background_color="#0e1117",
                colormap="hot",
                max_words=150,
            ).generate(neg_text)

            fig_wc_neg, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc_neg.to_array(), interpolation="bilinear")
            ax.axis("off")
            fig_wc_neg.patch.set_facecolor("#0e1117")
            st.pyplot(fig_wc_neg)
            plt.close(fig_wc_neg)
        except Exception as e:
            st.warning(f"Could not generate word cloud: {e}")
    else:
        st.info("No negative review text available.")



# 5. HYPOTHESIS TESTING — CHI-SQUARED

st.markdown('<div class="section-header"><h2>🧪 Hypothesis Testing (Chi-Squared)</h2></div>', unsafe_allow_html=True)

# --- Test 1: Sentiment vs Topic Cluster ---
st.subheader("Test 1: Sentiment vs Topic Cluster")
st.markdown("""
- **H₀ (Null):** Sentiment is independent of topic cluster — complaints are randomly distributed.
- **H₁ (Alternative):** Sentiment depends on the topic cluster — certain topics attract more negativity.
""")

with st.spinner("Running K-Means clustering for hypothesis test..."):
    cluster_sample = df.dropna(subset=["clean_text"]).sample(min(50000, len(df)), random_state=42)

    tfidf_cluster = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
    X_cluster = tfidf_cluster.fit_transform(cluster_sample["clean_text"])

    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    cluster_sample["cluster"] = kmeans.fit_predict(X_cluster)

    contingency_table = pd.crosstab(cluster_sample["cluster"], cluster_sample["target"])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

col_res1, col_heat1 = st.columns([1, 2])

with col_res1:
    st.markdown(f"""
    <div class="stat-box">
        <h4 style="color:#e94560;">Results</h4>
        <p><b>χ² Statistic:</b> <code>{chi2:,.2f}</code></p>
        <p><b>P-value:</b> <code>{p_value:.2e}</code></p>
        <p><b>Degrees of Freedom:</b> <code>{dof}</code></p>
        <p><b>Significance (α=0.05):</b>
            {"<span style='color:#00d2ff'>✅ REJECT H₀ — Statistically Significant</span>" if p_value < 0.05
            else "<span style='color:#e94560'>❌ FAIL TO REJECT H₀</span>"}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_heat1:
    fig_heat, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Observed Frequencies", fontsize=12, color="white")

    expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
    sns.heatmap(expected_df.round(0), annot=True, fmt=".0f", cmap="Oranges", ax=axes[1])
    axes[1].set_title("Expected Frequencies", fontsize=12, color="white")

    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    fig_heat.patch.set_facecolor("#0e1117")
    plt.tight_layout()
    st.pyplot(fig_heat)
    plt.close()

# --- Test 2: Sentiment vs Time Period ---
st.subheader("Test 2: Sentiment vs Time Period")
st.markdown("""
- **H₀ (Null):** Sentiment distribution is the same across all time periods — no real "spikes."
- **H₁ (Alternative):** Sentiment shifts significantly over time — crisis spikes are real.
""")

df_time_test = df.dropna(subset=["new_date"]).copy()
df_time_test["week"] = df_time_test["new_date"].dt.isocalendar().week.astype(int)

time_table = pd.crosstab(df_time_test["week"], df_time_test["target"])
chi2_t, p_val_t, dof_t, exp_t = chi2_contingency(time_table)

col_res2, col_chart2 = st.columns([1, 2])

with col_res2:
    st.markdown(f"""
    <div class="stat-box">
        <h4 style="color:#e94560;">Results</h4>
        <p><b>χ² Statistic:</b> <code>{chi2_t:,.2f}</code></p>
        <p><b>P-value:</b> <code>{p_val_t:.2e}</code></p>
        <p><b>Degrees of Freedom:</b> <code>{dof_t}</code></p>
        <p><b>Significance (α=0.05):</b>
            {"<span style='color:#00d2ff'>✅ REJECT H₀ — Sentiment spikes are real</span>" if p_val_t < 0.05
            else "<span style='color:#e94560'>❌ FAIL TO REJECT H₀ — No significant shift</span>"}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_chart2:
    weekly_sentiment = df_time_test.groupby(["week", "label"]).size().reset_index(name="count")
    fig_weekly = px.line(
        weekly_sentiment,
        x="week",
        y="count",
        color="label",
        color_discrete_map={"Positive": "#00d2ff", "Negative": "#e94560"},
        markers=True,
        labels={"count": "Reviews", "week": "Week Number"},
    )
    fig_weekly.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
    )
    st.plotly_chart(fig_weekly, use_container_width=True)


# 6. TOPIC CLUSTER ANALYSIS

st.markdown('<div class="section-header"><h2>🔬 Topic Cluster Analysis</h2></div>', unsafe_allow_html=True)

col_cluster_dist, col_cluster_sent = st.columns(2)

with col_cluster_dist:
    cluster_dist = cluster_sample["cluster"].value_counts().sort_index().reset_index()
    cluster_dist.columns = ["Cluster", "Count"]

    fig_cd = px.bar(
        cluster_dist,
        x="Cluster",
        y="Count",
        color="Count",
        color_continuous_scale="Viridis",
        labels={"Count": "Number of Reviews"},
    )
    fig_cd.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Cluster Distribution",
        height=400,
    )
    st.plotly_chart(fig_cd, use_container_width=True)

with col_cluster_sent:
    cluster_sent = pd.crosstab(cluster_sample["cluster"], cluster_sample["label"])

    fig_cs = px.imshow(
        cluster_sent,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        labels=dict(x="Sentiment", y="Cluster", color="Count"),
    )
    fig_cs.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Sentiment by Cluster",
        height=400,
    )
    st.plotly_chart(fig_cs, use_container_width=True)

# Show top terms per cluster
st.subheader("📝 Top Terms per Cluster")
terms = tfidf_cluster.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

cluster_cols = st.columns(6)
for i in range(6):
    with cluster_cols[i]:
        top_terms = [terms[ind] for ind in order_centroids[i, :8]]
        st.markdown(f"**Cluster {i}**")
        for t in top_terms:
            st.markdown(f"- {t}")



# 7. CRISIS REPORTS


st.markdown('<div class="section-header"><h2>🚨 Crisis Reports</h2></div>', unsafe_allow_html=True)

crisis_df = load_crisis_reports()

if crisis_df.empty:
    st.info("No crisis reports found in the database. Run the crisis pipeline first.")
else:
    for _, row in crisis_df.iterrows():
        with st.expander(f"🗓️ Crisis Report — {row.get('timestamp', 'N/A')}", expanded=False):
            st.markdown(row.get("summary", "No summary available."))



# 8. ENTITY INTELLIGENCE (NER)


st.markdown('<div class="section-header"><h2>🏷️ Entity Intelligence (NER)</h2></div>', unsafe_allow_html=True)

entity_df = load_entities()

if entity_df.empty:
    st.info("No entity data found. Run the NER pipeline first.")
else:
    col_ent_chart, col_ent_table = st.columns([2, 1])

    with col_ent_chart:
        fig_ent = px.bar(
            entity_df.head(15),
            x="mentions",
            y="entity",
            color="label",
            orientation="h",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"mentions": "Mentions", "entity": "Entity", "label": "NER Type"},
        )
        fig_ent.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"),
            height=450,
            title="Top Mentioned Entities",
        )
        st.plotly_chart(fig_ent, use_container_width=True)

    with col_ent_table:
        st.dataframe(
            entity_df,
            use_container_width=True,
            height=450,
        )


# 9. BRAND ASSISTANT CHATBOT (NLP-to-SQL)


st.markdown('<div class="section-header"><h2>🤖 Brand Assistant</h2></div>', unsafe_allow_html=True)
st.markdown("Ask questions about brand health in plain English. The assistant translates your query into SQL.")

# --- NLP-to-SQL Query Mapper ---
def parse_question_to_sql(question: str) -> tuple:
    """Parse a natural language question into a SQL query and description.

    Uses keyword matching to identify intent and map to predefined SQL queries.
    Returns (sql_query, description) tuple.
    """
    q = question.lower().strip()

    # --- Complaint / Negative queries ---
    if any(w in q for w in ["complaint", "negative", "worst", "bad", "problem", "issue", "crisis"]):

        if any(w in q for w in ["top", "most", "common", "frequent"]):
            # Extract number if mentioned (default 5)
            import re
            num_match = re.search(r'\b(\d+)\b', q)
            n = int(num_match.group(1)) if num_match else 5
            return (
                text("""
                    SELECT clean_text, new_date
                    FROM fact_reviews
                    WHERE sentiment = 0
                    ORDER BY new_date DESC
                    LIMIT :n
                """),
                {"n": n},
                f"Top {n} most recent complaints"
            )

        if any(w in q for w in ["today", "24 hour", "last day", "recent"]):
            return (
                text("""
                    SELECT clean_text, new_date
                    FROM fact_reviews
                    WHERE sentiment = 0
                    ORDER BY new_date DESC
                    LIMIT 10
                """),
                {},
                "Most recent negative reviews"
            )

        if "count" in q or "how many" in q:
            return (
                text("SELECT COUNT(*) AS total_complaints FROM fact_reviews WHERE sentiment = 0"),
                {},
                "Total number of negative reviews"
            )

        return (
            text("""
                SELECT clean_text, new_date
                FROM fact_reviews
                WHERE sentiment = 0
                ORDER BY new_date DESC
                LIMIT 10
            """),
            {},
            "Latest negative reviews"
        )

    # --- Positive queries ---
    if any(w in q for w in ["positive", "good", "best", "happy", "great", "love"]):
        if "count" in q or "how many" in q:
            return (
                text("SELECT COUNT(*) AS total_positive FROM fact_reviews WHERE sentiment = 4"),
                {},
                "Total number of positive reviews"
            )
        return (
            text("""
                SELECT clean_text, new_date
                FROM fact_reviews
                WHERE sentiment = 4
                ORDER BY new_date DESC
                LIMIT 10
            """),
            {},
            "Latest positive reviews"
        )

    # --- Sentiment / Health Score ---
    if any(w in q for w in ["sentiment", "health", "nss", "score", "overall", "brand"]):
        return (
            text("""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN sentiment = 4 THEN 1 ELSE 0 END) AS positive,
                    SUM(CASE WHEN sentiment = 0 THEN 1 ELSE 0 END) AS negative,
                    ROUND(
                        (SUM(CASE WHEN sentiment = 4 THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100) -
                        (SUM(CASE WHEN sentiment = 0 THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100), 2
                    ) AS net_sentiment_score
                FROM fact_reviews
            """),
            {},
            "Overall brand health — Net Sentiment Score"
        )

    # --- Entity queries ---
    if any(w in q for w in ["entity", "entities", "ner", "person", "organization", "company", "who"]):
        return (
            text("""
                SELECT entity, label, COUNT(*) AS mentions
                FROM entity_store
                GROUP BY entity, label
                ORDER BY mentions DESC
                LIMIT 10
            """),
            {},
            "Top mentioned entities from NER analysis"
        )

    # --- Crisis queries ---
    if any(w in q for w in ["crisis", "alert", "report", "summary", "brief"]):
        return (
            text("SELECT id, timestamp, summary FROM crisis_reports ORDER BY timestamp DESC LIMIT 5"),
            {},
            "Latest crisis reports"
        )

    # --- Time / Trend queries ---
    if any(w in q for w in ["trend", "time", "week", "month", "over time", "change"]):
        return (
            text("""
                SELECT
                    DATE_TRUNC('week', new_date) AS week,
                    SUM(CASE WHEN sentiment = 4 THEN 1 ELSE 0 END) AS positive,
                    SUM(CASE WHEN sentiment = 0 THEN 1 ELSE 0 END) AS negative
                FROM fact_reviews
                WHERE new_date IS NOT NULL
                GROUP BY week
                ORDER BY week
            """),
            {},
            "Weekly sentiment trend"
        )

    # --- Count / Total ---
    if any(w in q for w in ["total", "count", "how many", "number"]):
        return (
            text("SELECT COUNT(*) AS total_reviews FROM fact_reviews"),
            {},
            "Total number of reviews in the database"
        )

    # --- Fallback ---
    return (
        text("""
            SELECT
                COUNT(*) AS total_reviews,
                SUM(CASE WHEN sentiment = 4 THEN 1 ELSE 0 END) AS positive,
                SUM(CASE WHEN sentiment = 0 THEN 1 ELSE 0 END) AS negative
            FROM fact_reviews
        """),
        {},
        "General overview — I wasn't sure what you meant, so here's a summary"
    )


# --- Chat Interface ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Chat input
user_question = st.chat_input("Ask about brand health... e.g. 'What are the top 3 complaints?'")

if user_question:
    st.chat_message("user").markdown(user_question)
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    try:
        query, params, description = parse_question_to_sql(user_question)

        engine = get_engine()
        result_df = pd.read_sql(query, engine, params=params)

        response = f"**📋 {description}**\n\n"
        response += f"*SQL Intent: {description}*\n\n"

        if result_df.empty:
            response += "No results found for this query."
        elif len(result_df.columns) <= 3 and len(result_df) == 1:
            # Single-row result — show as metrics
            for col in result_df.columns:
                val = result_df[col].iloc[0]
                if isinstance(val, (int, float)):
                    response += f"- **{col.replace('_', ' ').title()}**: `{val:,.2f}`\n"
                else:
                    response += f"- **{col.replace('_', ' ').title()}**: {val}\n"
        else:
            response += result_df.to_markdown(index=False)

        st.chat_message("assistant").markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    except Exception as e:
        error_msg = f"❌ Error processing query: {e}"
        st.chat_message("assistant").markdown(error_msg)
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})


# Example questions
with st.expander("💡 Example questions you can ask"):
    st.markdown("""
    - What are the top 3 complaints?
    - How many negative reviews are there?
    - Show me the latest positive reviews
    - What is the overall brand health score?
    - Who are the top mentioned entities?
    - Show me the latest crisis reports
    - What is the sentiment trend over time?
    - How many total reviews do we have?
    """)



# FOOTER


st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#a8b2d1;'>"
    "VoxPop Intelligence Dashboard • Built with Streamlit & Plotly • Task 5 Deployment"
    "</p>",
    unsafe_allow_html=True,
)
