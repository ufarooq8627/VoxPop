VoxPop – AI-Driven Crisis Intelligence Pipeline
Overview

VoxPop detects customer crises from reviews using classical ML, deep learning, and transformer models. It generates crisis summaries, extracts entities, and stores insights in PostgreSQL for analytics.

Task 1 — Data Warehouse & JSON Pipeline

Star Schema Warehouse Design

A relational star schema structures social sentiment data:

Fact Table

fact_reviews → cleaned text, sentiment, timestamp, user_id

Dimension Tables

dim_users → unique users

dim_sentiment → sentiment categories

location dimension reserved for expansion

This enables fast joins, aggregations, and time-series analytics with normalized integrity.

JSON Processing Pipeline

Raw JSON ingestion

Cleaning & normalization

Structured warehouse loading

Task 2 — Classical ML Baseline

Feature Engineering

TF-IDF vectorization

Text normalization

Train/test split

Supervised Model

Logistic Regression classifier

Accuracy + confusion matrix

Unsupervised Analysis

K-Means topic clustering

Cluster–sentiment correlation

Business interpretation

Outcome
Interpretable baseline with topic insights.

Task 3 — Deep Learning Emotional Modeling

LSTM Pipeline

Vocabulary building

Integer encoding

Sequence padding

PyTorch tensors

GPU training

Model

Embedding + LSTM

Sigmoid binary output

Training

Mini-batch DataLoader

Adam optimizer

BCELoss

Performance

Accuracy ≈ 83%

F1 ≈ 0.83

Balanced precision/recall

Task 4 — Transformer Crisis Intelligence

Complaint detection (RoBERTa)

Topic grouping

Crisis summarization (BART)

Named Entity Recognition (BERT NER)

PostgreSQL entity storage

SQL analytics queries

Tech Stack

Python • PyTorch • Hugging Face • PostgreSQL • SQLAlchemy • Jupyter

How to Run
python voxpop.py


or run notebook cells.
