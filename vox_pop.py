# %%
!pip install gensim
!pip install --upgrade pandas

# %%
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
import psycopg2
from sqlalchemy import create_engine, Column, Integer, String, Date, ForeignKey, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from collections import Counter
from sklearn.linear_model import LogisticRegression 

# %% [markdown]
# ## Task - 1

# %%
cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv('training.1600000.processed.noemoticon.csv',
                        header=None, names=cols,encoding="latin-1")
df.info()

# %%
# Extracts the timezone part (3-4 uppercase letters) from the string
timezones = df['date'].str.extract(r'([A-Z]{3,4})')[0].unique()

print("Timezones found:", timezones)

# Counts occurrences of each timezone abbreviation
tz_counts = df['date'].str.extract(r'([A-Z]{3,4})')[0].value_counts()

print(tz_counts)
# this ensures that all the pdt is the only timezone 
# and the coutries fall under those timezones can be used to make dim_locations

# %%
df['date'] = df['date'].str.replace(r'\b[A-Z]{3}\b', '', regex=True)

df['new_date'] = pd.to_datetime(
    df['date'],
    format='%a %b %d %H:%M:%S %Y',
    errors='coerce'
)


# %%
print(df[['date', 'new_date']].head())
print(df['new_date'].isna().sum())


# %%
def clean_data(text, keep_numbers=False):

    if not isinstance(text, str):
        return ""

    # normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # remove emails
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)

    # remove mentions
    text = re.sub(r'@\w+', ' ', text)

    # remove urls
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # convert hashtags to just the word (keep the token)
    text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)

    # keep alphanum and spaces (optionally keep numbers)
    if keep_numbers:
        text = re.sub(r'[^0-9A-Za-z\s]', ' ', text)
    else:
        text = re.sub(r'[^A-Za-z\s]', ' ', text)

    # collapse repeated punctuation/characters and whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # lowercase
    text = text.lower()

    return text

# %%
df['clean_text'] = df['text'].apply(clean_data)
df['target'] = df['sentiment'].apply(lambda x: 1 if x == 4 else 0)
df.head()

# %%
df.info()

# %%
print(df.duplicated().sum())
print(df.isna().sum())

# %%
# pip install psycopg2-binary

# %%
# connecting wil postgres

engine = create_engine(
    "postgresql+psycopg://postgres:Greybentley%40123@127.0.0.1:5432/voxpop"
)

# %%
try:
    with engine.connect() as conn:
        print("✅ SQL Alchemy connected to Vox Pop")
except Exception as e:
    print("❌ Connection failed:", e)

# %%

# drop_tables_sql = """
# DROP TABLE IF EXISTS fact_reviews CASCADE;
# DROP TABLE IF EXISTS dim_users CASCADE;
# DROP TABLE IF EXISTS dim_sentiment CASCADE;
# """

# with engine.begin() as conn:
#     conn.execute(text(drop_tables_sql))

# print("🗑️ Existing tables deleted (if they existed)")

# %%
#  DROP TABLES IN SAFE ORDER
from sqlalchemy import text

with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS fact_reviews;"))
    conn.execute(text("DROP TABLE IF EXISTS dim_users;"))
    conn.execute(text("DROP TABLE IF EXISTS dim_sentiment;"))

print(" Old tables dropped")


#  DIM USERS
dim_users = df[['user']].drop_duplicates().reset_index(drop=True)
dim_users['user_id'] = dim_users.index + 1

dim_users.to_sql(
    'dim_users',
    con=engine,
    if_exists='replace',
    index=False
)

print(" dim_users loaded")


# 2. FACT PREP
df_merged = df.merge(dim_users, on='user', how='left')

df_merged['new_date'] = pd.to_datetime(df_merged['new_date'], errors='coerce')

fact_reviews = df_merged[['id', 'clean_text', 'sentiment', 'new_date', 'user_id']]
fact_reviews = fact_reviews.rename(columns={'id': 'review_id'})
fact_reviews = fact_reviews.drop_duplicates(subset=['review_id'])


#  FACT TABLE
fact_reviews.to_sql(
    'fact_reviews',
    con=engine,
    if_exists='replace',
    index=False,
    chunksize=5000
)

print(" fact_reviews loaded")


#  DIM SENTIMENT
sentiment_data = {
    'sentiment_id': [0, 4],
    'label_name': ['Negative', 'Positive']
}

dim_sentiment = pd.DataFrame(sentiment_data)

dim_sentiment.to_sql(
    'dim_sentiment',
    con=engine,
    if_exists='replace',
    index=False
)

print(" dim_sentiment loaded")
print(" Star Schema done")

# %%
# As we do not have have location data so I am going to skip it.(Dim_location skipped)

# %% [markdown]
# ## Task - 2

# %% [markdown]
# ##### TFIDF + Log Reg(Classical ML baseline model using Supervised learning) 

# %%
sentiment_counts = df['sentiment'].value_counts()

# %%
# spliting the data
X = df['clean_text']
y = df['target'] # target column is sentiment column

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# training the model
tfidf = TfidfVectorizer(max_features=20000, stop_words='english', ngram_range=(1,2))

X_train_vect = tfidf.fit_transform(X_train)
X_test_vect = tfidf.transform(X_test)


# %%
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_vect, y_train)


# %%
# testing the model
y_pred = log_reg.predict(X_test_vect)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# confusion metrics
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('What the Robot Predicted')
plt.ylabel('What was Actually True')
plt.title('The Judge\'s Scorecard (Confusion Matrix)')
plt.show()

# %%
# In Task 2 I built a classical machine learning baseline.
# First I converted text into numeric features using TF-IDF.
# Then I trained a Logistic Regression classifier to predict sentiment.
# I split the dataset into train and test to avoid data leakage.
# After training, I evaluated the model using accuracy, precision, recall, F1 score and a confusion matrix.


# %% [markdown]
# Un-Supervised Learning

# %%
sample_df = df.sample(50000, random_state=42)

# %%
tfidf_cluster = TfidfVectorizer(
    max_features=20000,
    stop_words='english',
    ngram_range=(1,2)
)

X_cluster = tfidf_cluster.fit_transform(sample_df['clean_text'])


# %%
inertia = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, 'bx-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# %%
# The curve clearly bends at 6 after it improvements are minor.

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans.fit(X_cluster)

# %%
terms = tfidf_cluster.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

for i in range(6):
    print(f"\nCluster {i}:")
    for ind in order_centroids[i, :8]:
        print(terms[ind])


# %%
X_all_cluster = tfidf_cluster.transform(df['clean_text'])
df['cluster'] = kmeans.predict(X_all_cluster)


# %%
pd.crosstab(df['cluster'], df['sentiment'])


# %%
sns.countplot(x='cluster', data=df)
plt.title('Cluster distribution')
plt.show()


# %%
cluster_sent = pd.crosstab(df['cluster'], df['sentiment'])

sns.heatmap(cluster_sent, annot=True, fmt='d')
plt.title('Sentiment by Cluster')
plt.show()


# %%
# In Task 2 I built a supervised baseline using TF-IDF and Logistic Regression to predict sentiment.
#  Then I used K-Means clustering to discover hidden topics.
#  By comparing clusters with sentiment, I identified which topics generate complaints versus positive feedback.

# %%
# The dataset did not contain geographical information, so regional EDA could not be performed. 
# If location data becomes available, the same pivot-table analysis can be applied.

# %% [markdown]
# ## Task - 3

# %%
# Text → sequence → LSTM → emotion score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%
from collections import Counter

words = " ".join(df['clean_text']).split()
vocab = Counter(words)

vocab_to_int = {word:i+1 for i,(word,_) in enumerate(vocab.most_common(20000))}


# %%
def encode(text):
    return [vocab_to_int[word] for word in text.split() if word in vocab_to_int]

df['encoded'] = df['clean_text'].apply(encode)


# %%
max_len = 50
def pad(seq, max_len):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [0] * (max_len - len(seq))

X_seq = np.array([pad(s, max_len) for s in df['encoded']])


# %%


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, df['target'],
    test_size=0.2,
    random_state=42
)


# %%
import torch

X_train = torch.tensor(X_train, dtype=torch.long).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)

X_test = torch.tensor(X_test, dtype=torch.long).to(device)
y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)


# %%
from torch.utils.data import TensorDataset, DataLoader

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

# %%
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)


# %%


# %%
model = LSTMModel(len(vocab_to_int)+1).to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        preds = model(xb)
        loss = loss_fn(preds.squeeze(), yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(train_loader):.4f}")



# %%
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)


# %%
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)

        preds = model(xb)
        preds = (preds.squeeze() > 0.5).float()

        correct += (preds == yb).sum().item()
        total += yb.size(0)

accuracy = correct / total
print("Test Accuracy:", accuracy)


# %%
from sklearn.metrics import classification_report

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb)
        preds = (preds.squeeze() > 0.5).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(yb.cpu().numpy())

print(classification_report(all_labels, all_preds))


# %%
torch.cuda.empty_cache()


# %%
# Deep Learning Emotional Modeling

# A PyTorch LSTM neural network was implemented to capture emotional patterns in sequential text. 
# Reviews were encoded into padded numeric sequences and trained using GPU-accelerated mini-batch gradient descent.

# Training loss consistently decreased, demonstrating stable learning.
#  The final model achieved approximately 83% accuracy and F1-score on unseen test data, with balanced precision and recall across both sentiment classes.

# Batched inference was used during evaluation to prevent GPU memory overflow and ensure scalable deployment. 
# The results confirm that sequence-based deep learning models outperform classical bag-of-words baselines by capturing contextual emotional signals.

# %% [markdown]
# ## Task - 4

# %%
!python -m pip install --upgrade transformers accelerate sentencepiece

# %%
import transformers
print(transformers.__version__)


# %%
import sys
print(sys.executable)


# %%
from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "facebook/bart-large-cnn"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

clf_name = "cardiffnlp/twitter-roberta-base-sentiment"

clf_tokenizer = AutoTokenizer.from_pretrained(clf_name)
clf_model = AutoModelForSequenceClassification.from_pretrained(clf_name)
clf_model.to(device)

# %%
# from sqlalchemy import create_engine, text

# engine = create_engine(
#     "postgresql+psycopg://postgres:Greybentley%40123@127.0.0.1:5432/voxpop"
# )


# query = text("""
# SELECT table_name
# FROM information_schema.tables
# WHERE table_schema = 'public';
# """)

# with engine.connect() as conn:
#     tables = conn.execute(query).fetchall()

# print(tables)



# %%
from sqlalchemy import create_engine, text
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import torch
import torch.nn.functional as F
from datetime import datetime

from transformers import AutoModelForTokenClassification
from transformers import pipeline as hf_pipeline

#  Load NER model
ner_name = "dslim/bert-base-NER"

ner_tokenizer = AutoTokenizer.from_pretrained(ner_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_name)

ner = hf_pipeline(
    "ner",
    model=ner_model,
    tokenizer=ner_tokenizer,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1

)



# CONFIG 
MAX_CHUNK_CHARS = 2500

engine = create_engine(
    "postgresql+psycopg://postgres:Greybentley%40123@127.0.0.1:5432/voxpop"
)

#  Load BART
model_name = "facebook/bart-large-cnn"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#  Load complaint classifier 
clf_name = "cardiffnlp/twitter-roberta-base-sentiment"

clf_tokenizer = AutoTokenizer.from_pretrained(clf_name)
clf_model = AutoModelForSequenceClassification.from_pretrained(clf_name)
clf_model.to(device)


def is_complaint(text_):
    inputs = clf_tokenizer(
        text_,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = clf_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)

    return probs[0][0] > 0.7


# Topic grouping
def group_by_topic(reviews):
    topics = {
        "browser": [],
        "mobile": [],
        "website": [],
        "system": [],
        "other": []
    }

    for r in reviews:
        t = r.lower()

        if "firefox" in t or "browser" in t:
            topics["browser"].append(r)

        elif "phone" in t or "mobile" in t or "app" in t:
            topics["mobile"].append(r)

        elif "website" in t or "server" in t:
            topics["website"].append(r)

        elif "windows" in t or "driver" in t or "computer" in t:
            topics["system"].append(r)

        else:
            topics["other"].append(r)

    return topics


def is_business_issue(text_):
    keywords = [
        "app", "website", "server", "login", "payment",
        "error", "bug", "crash", "down", "support",
        "account", "update", "device", "service"
    ]

    t = text_.lower()

    return any(k in t for k in keywords)


#  Fetch negative reviews
def fetch_negative_reviews(limit=1000):
    query = text("""
        SELECT clean_text
        FROM fact_reviews
        WHERE sentiment = 0
        AND LENGTH(clean_text) > 40
        LIMIT :limit
    """)

    with engine.connect() as conn:
        rows = conn.execute(query, {"limit": limit}).fetchall()

    reviews = list(set([r[0] for r in rows if r[0]]))

    filtered = []

    print("Running complaint classifier...")

    for text_ in reviews:
        if is_complaint(text_) and is_business_issue(text_):
            filtered.append(text_)


    return filtered


#  Chunk long text
def chunk_text(text, chunk_size=MAX_CHUNK_CHARS):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


#  Summarize single chunk
def summarize_text(text):
    inputs = tokenizer(
        text,
        max_length=1024,
        return_tensors="pt",
        truncation=True
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=120,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Summarize all chunks
def summarize_chunks(chunks):
    summaries = []

    for i, chunk in enumerate(chunks, 1):
        print(f"Summarizing chunk {i}/{len(chunks)}...")
        summaries.append(summarize_text(chunk))

    return " ".join(summaries)


# Executive compression
def compress_summary(text):
    print("Running executive compression...")

    inputs = tokenizer(
        text,
        max_length=1024,
        return_tensors="pt",
        truncation=True
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=6,
        max_length=80,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Store crisis report
def store_summary(summary_text):
    create_table = text("""
        CREATE TABLE IF NOT EXISTS crisis_reports (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP,
            summary TEXT
        )
    """)

    insert_query = text("""
        INSERT INTO crisis_reports (timestamp, summary)
        VALUES (:timestamp, :summary)
        RETURNING id
    """)

    with engine.begin() as conn:
        conn.execute(create_table)
        result = conn.execute(insert_query, {
            "timestamp": datetime.now(),
            "summary": summary_text
        })

        crisis_id = result.scalar()

    return crisis_id


def store_entities(summary_text, crisis_id):
    create_table = text("""
        CREATE TABLE IF NOT EXISTS entity_store (
            id SERIAL PRIMARY KEY,
            crisis_id INTEGER,
            entity TEXT,
            label TEXT
        )
    """)

    entities = ner(summary_text)

    with engine.begin() as conn:
        conn.execute(create_table)

        for ent in entities:
            conn.execute(text("""
                INSERT INTO entity_store (crisis_id, entity, label)
                VALUES (:cid, :ent, :lab)
            """), {
                "cid": crisis_id,
                "ent": ent["word"],
                "lab": ent["entity_group"]
            })


# Main pipeline
def run_crisis_summary():
    print("Fetching negative reviews...")
    reviews = fetch_negative_reviews()

    if not reviews:
        print("No negative reviews found.")
        return

    topics = group_by_topic(reviews)

    final_reports = []

    for name, texts in topics.items():
        if not texts:
            continue

        combined = " ".join(texts)
        chunks = chunk_text(combined)

        print(f"\nTopic: {name.upper()} | chunks: {len(chunks)}")

        summary = summarize_chunks(chunks)
        summary = compress_summary(summary)

        final_reports.append(f"[{name.upper()}] {summary}")

    final_summary = "\n".join(final_reports)

    
    crisis_id = store_summary(final_summary)
    store_entities(final_summary, crisis_id)

    print("\nCrisis Summary Generated:\n")
    print(final_summary)


#  Execute 
if __name__ == "__main__":
    run_crisis_summary()


# %%
from sqlalchemy import create_engine, text
from transformers import ( BartTokenizer, BartForConditionalGeneration, AutoTokenizer, 
                          AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline as hf_pipeline)
import torch
import torch.nn.functional as F
from datetime import datetime

# CONFIG 
MAX_CHUNK_CHARS = 2500

engine = create_engine(
    "postgresql+psycopg://postgres:Greybentley%40123@127.0.0.1:5432/voxpop"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BART summarizer 
model_name = "facebook/bart-large-cnn"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model.to(device)

# Load complaint classifier
clf_name = "cardiffnlp/twitter-roberta-base-sentiment"

clf_tokenizer = AutoTokenizer.from_pretrained(clf_name)
clf_model = AutoModelForSequenceClassification.from_pretrained(clf_name)
clf_model.to(device)

#  Load NER model
ner_name = "dslim/bert-base-NER"

ner_tokenizer = AutoTokenizer.from_pretrained(ner_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_name)

ner = hf_pipeline(
    "ner",
    model=ner_model,
    tokenizer=ner_tokenizer,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1
)


# %%
def is_complaint(text_):
    inputs = clf_tokenizer(
        text_,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = clf_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)

    return probs[0][0] > 0.7


def group_by_topic(reviews):
    topics = {
        "browser": [],
        "mobile": [],
        "website": [],
        "system": [],
        "other": []
    }

    for r in reviews:
        t = r.lower()

        if "firefox" in t or "browser" in t:
            topics["browser"].append(r)

        elif "phone" in t or "mobile" in t or "app" in t:
            topics["mobile"].append(r)

        elif "website" in t or "server" in t:
            topics["website"].append(r)

        elif "windows" in t or "driver" in t or "computer" in t:
            topics["system"].append(r)

        else:
            topics["other"].append(r)

    return topics


def is_business_issue(text_):
    keywords = [
        "app", "website", "server", "login", "payment",
        "error", "bug", "crash", "down", "support",
        "account", "update", "device", "service"
    ]

    t = text_.lower()
    return any(k in t for k in keywords)


def fetch_negative_reviews(limit=1000):
    query = text("""
        SELECT clean_text
        FROM fact_reviews
        WHERE sentiment = 0
        AND LENGTH(clean_text) > 40
        LIMIT :limit
    """)

    with engine.connect() as conn:
        rows = conn.execute(query, {"limit": limit}).fetchall()

    reviews = list(set([r[0] for r in rows if r[0]]))

    filtered = []

    print("Running complaint classifier...")

    for text_ in reviews:
        if is_complaint(text_) and is_business_issue(text_):
            filtered.append(text_)

    return filtered


def chunk_text(text, chunk_size=MAX_CHUNK_CHARS):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def summarize_text(text):
    inputs = tokenizer(
        text,
        max_length=1024,
        return_tensors="pt",
        truncation=True
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=120,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_chunks(chunks):
    summaries = []

    for i, chunk in enumerate(chunks, 1):
        print(f"Summarizing chunk {i}/{len(chunks)}...")
        summaries.append(summarize_text(chunk))

    return " ".join(summaries)


def compress_summary(text):
    print("Running executive compression...")

    inputs = tokenizer(
        text,
        max_length=1024,
        return_tensors="pt",
        truncation=True
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=6,
        max_length=80,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def store_summary(summary_text):
    create_table = text("""
        CREATE TABLE IF NOT EXISTS crisis_reports (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP,
            summary TEXT
        )
    """)

    insert_query = text("""
        INSERT INTO crisis_reports (timestamp, summary)
        VALUES (:timestamp, :summary)
        RETURNING id
    """)

    with engine.begin() as conn:
        conn.execute(create_table)
        result = conn.execute(insert_query, {
            "timestamp": datetime.now(),
            "summary": summary_text
        })

        crisis_id = result.scalar()

    return crisis_id


def store_entities(summary_text, crisis_id):
    create_table = text("""
        CREATE TABLE IF NOT EXISTS entity_store (
            id SERIAL PRIMARY KEY,
            crisis_id INTEGER,
            entity TEXT,
            label TEXT
        )
    """)

    entities = ner(summary_text)

    with engine.begin() as conn:
        conn.execute(create_table)

        for ent in entities:
            conn.execute(text("""
                INSERT INTO entity_store (crisis_id, entity, label)
                VALUES (:cid, :ent, :lab)
            """), {
                "cid": crisis_id,
                "ent": ent["word"],
                "lab": ent["entity_group"]
            })


# %%
def run_crisis_summary():
    print("Fetching negative reviews...")
    reviews = fetch_negative_reviews()

    if not reviews:
        print("No negative reviews found.")
        return

    topics = group_by_topic(reviews)
    final_reports = []

    for name, texts in topics.items():
        if not texts:
            continue

        combined = " ".join(texts)
        chunks = chunk_text(combined)

        print(f"\nTopic: {name.upper()} | chunks: {len(chunks)}")

        summary = summarize_chunks(chunks)
        summary = compress_summary(summary)

        final_reports.append(f"[{name.upper()}] {summary}")

    final_summary = "\n".join(final_reports)

    crisis_id = store_summary(final_summary)
    store_entities(final_summary, crisis_id)

    print("\nCrisis Summary Generated:\n")
    print(final_summary)


run_crisis_summary()


# %%
# The crisis summary reflects the dominant negative signals present in the dataset. 
# Since the dataset contains general emotional distress and technical frustrations rather 
# than brand-specific complaints, the generated report captures personal and technical crisis patterns 
# instead of product failures.


# %% [markdown]
# NER code

# %%
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline as ner_pipeline


# %%
ner_name = "dslim/bert-base-NER"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_name)
ner = ner_pipeline(
    "ner",
    model=ner_model,
    tokenizer=ner_tokenizer,
    aggregation_strategy="simple"
)

# %%
# for checking purpose whether NER is working or not?
summary_text = """
Users report that the Apple iPhone app crashes in New York.
Google support has not responded since Monday.
CEO Tim Cook was mentioned in complaints.
"""


entities = ner(summary_text)

for e in entities:
    print(e)

# %% [markdown]
# SQL Part
# 

# %%
# Top mentioned entities

query = text("""
SELECT entity, label, COUNT(*) AS mentions
FROM entity_store
GROUP BY entity, label
ORDER BY mentions DESC
LIMIT 10;
""")

df = pd.read_sql(query, engine)
df


# %%
# Crisis history timeline

query = text("""
SELECT DATE(timestamp) AS day, COUNT(*) AS reports
FROM crisis_reports
GROUP BY day
ORDER BY day;
""")

pd.read_sql(query, engine)



# %%
# Entities per crisis
query = text("""
SELECT crisis_id, COUNT(*) AS entity_count
FROM entity_store
GROUP BY crisis_id
ORDER BY entity_count DESC;
""")

pd.read_sql(query, engine)

# %%
def main():
    run_crisis_summary()

if __name__ == "__main__":
    main()







