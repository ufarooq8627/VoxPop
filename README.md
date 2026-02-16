# 📊 VoxPop: AI-Driven Global Brand Sentiment & Crisis Intelligence

VoxPop is an end-to-end AI platform that ingests raw social media data, stores it in a structured SQL warehouse, uses Deep Learning to detect emotional nuances and potential PR crises, and provides Generative AI-powered crisis reporting and entity intelligence.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-FF4B4B?logo=streamlit)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791?logo=postgresql)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)

---

## 🏗️ Architecture

```
Raw Tweets (1.6M) → Regex Cleaner → PostgreSQL Star Schema
                                          ↓
                        ┌─────────────────┼─────────────────┐
                        ↓                 ↓                 ↓
                  TF-IDF + LogReg    K-Means (6)       LSTM (PyTorch)
                  (Baseline)         (Topic Clusters)   (Emotion Score)
                        ↓                 ↓                 ↓
                        └─────────────────┼─────────────────┘
                                          ↓
                              BART Summarizer + BERT NER
                                          ↓
                              Streamlit Dashboard + Crisis Reports
```

---

## 🚀 Features

| Module | Description |
|--------|-------------|
| **Data Pipeline** | Regex-based text cleaner, PostgreSQL Star Schema (`fact_reviews`, `dim_users`, `dim_sentiment`) |
| **Classical ML** | TF-IDF + Logistic Regression baseline (**F1: 0.78, AUC: 0.86**) |
| **Topic Clustering** | K-Means (6 clusters) to group reviews into topics like Pricing, Quality, Shipping |
| **Deep Learning** | PyTorch LSTM for emotional intensity scoring (**F1: 0.83, Accuracy: 83%**) |
| **Summarization** | BART (`facebook/bart-large-cnn`) condenses 1000+ negative reviews into a 3-sentence Crisis Report |
| **Named Entity Recognition** | BERT (`dslim/bert-base-NER`) extracts people, organizations, and locations from reviews |
| **Hypothesis Testing** | Chi-Squared tests proving sentiment shifts are statistically significant (p < 0.05) |
| **Dashboard** | Interactive Streamlit dashboard with KPIs, word clouds, donut charts, and crisis alerts |

---

## 📂 Project Structure

```
Final_VoxPop/
├── vox_pop.py                  # Main pipeline 
├── voxpop_main.ipynb           # Jupyter notebook with outputs
├── streamlit_app.py            # Interactive dashboard
├── Data Ingestor.ipynb         # Dataset download script
├── draft.ipynb                 # Experimentation notebook
├── info.txt                    # Model comparison results
└── README.md
```

> **Note:** The dataset files (`.csv`, `.parquet`) and trained model weights (`.pt`, `.pth`) are excluded from this repo due to size limits. Run `Data Ingestor.ipynb` to download the Sentiment140 dataset automatically.


---

## 📊 Model Performance

| Model | F1-Score | AUC-ROC |
|-------|----------|---------|
| **TF-IDF + Logistic Regression** | **0.78** | **0.86** |
| W2V + Random Forest | — | 0.82 |
| W2V + XGBoost | 0.74 | 0.82 |
| W2V + Linear SVC | 0.74 | 0.81 |
| W2V + Gaussian NB | 0.61 | 0.68 |
| **PyTorch LSTM** | **0.83** | — |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- PostgreSQL 15+
- CUDA-compatible GPU (recommended)

### Install Dependencies

```bash
pip install pandas numpy scikit-learn torch transformers
pip install streamlit plotly matplotlib seaborn wordcloud
pip install sqlalchemy psycopg gensim scipy
```

### Database Setup

1. Create a PostgreSQL database named `voxpop`
2. Set your database connection string as an environment variable:
   ```bash
   # Windows (PowerShell)
   [System.Environment]::SetEnvironmentVariable("VOXPOP_DB_URL", "postgresql+psycopg://postgres:YOUR_PASSWORD@127.0.0.1:5432/voxpop", "User")

   # Linux / macOS
   export VOXPOP_DB_URL="postgresql+psycopg://postgres:YOUR_PASSWORD@127.0.0.1:5432/voxpop"
   ```
3. Run the pipeline to populate tables:
   ```bash
   python vox_pop.py
   ```

### Launch Dashboard

```bash
streamlit run streamlit_app.py
```

---

## 📈 Dashboard Sections

The Streamlit dashboard includes:

1. **KPI Metrics** — Total reviews, Positive/Negative counts, Net Sentiment Score (NSS)
2. **Sentiment Distribution** — Interactive donut chart
3. **Sentiment Over Time** — Stacked bar chart by week
4. **Word Clouds** — Separate clouds for positive and negative reviews
5. **Hypothesis Testing** — Chi-Squared results with observed vs expected heatmaps
6. **Topic Clusters** — Cluster distribution + top terms per cluster
7. **Crisis Reports** — Auto-generated crisis summaries from the database
8. **Entity Intelligence** — Top mentioned entities (people, orgs, locations)
9. **Brand Assistant Chatbot** — NLP-to-SQL interface for querying brand health in plain English

---

## 🧪 Hypothesis Testing

Two Chi-Squared tests validate that the AI's findings are statistically significant:

| Test | H₀ (Null Hypothesis) | Result |
|------|----------------------|--------|
| Sentiment vs Cluster | Complaints are randomly distributed across topics | **Rejected** (p < 0.05) |
| Sentiment vs Time | Sentiment is the same across all time periods | **Rejected** (p < 0.05) |

---

## 🧰 Tech Stack

- **Languages**: Python, SQL
- **ML/DL**: PyTorch, Scikit-Learn, HuggingFace Transformers
- **Models**: LSTM, BART, BERT, Logistic Regression, K-Means
- **Database**: PostgreSQL, SQLAlchemy, Star Schema
- **Visualization**: Streamlit, Plotly, Matplotlib, Seaborn, WordCloud
- **Statistics**: SciPy (Chi-Squared), Hypothesis Testing

---

## 📜 License

This project was developed as part of an AI/ML coursework assignment.

---

