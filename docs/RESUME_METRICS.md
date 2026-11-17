# Project metrics (verified on this repository)

| Metric | Value |
|--------|-------|
| Supported languages | 5 - English, Hindi, French, Spanish, German |
| Multilingual sample dataset | 105 samples, 7 categories, 7 response types |
| Evaluation approach | Hybrid rule-based + Gemini (weighted merge, automatic fallback) |
| Batch evaluation | CSV upload, per-row validation, concurrent Gemini calls |
| REST API | 16 endpoints (evaluation, batch, results, analytics, reports, health) |
| SQL database | 4 normalised tables, composite indexes, SQLite + PostgreSQL, connection pooling |
| Analytics APIs | 6 endpoints: overview, languages, categories, hallucination types, low-quality, trends |
| ML model | Random Forest quality classifier (4 classes): 92.6% test accuracy, 95.2% 5-fold CV accuracy* |
| Reports | CSV, Excel (formatted, summary sheet) and PDF exports |
| Visual analytics | 6 Matplotlib charts (distributions + correlation heatmap) |
| Security | API-key auth, sliding-window rate limiting, strict CORS, security headers, input sanitisation, CSV-injection protection |
| Automated tests | 217 pytest tests (API, database, Gemini mock, batch, analytics, reports, ML, security) |
| Delivery | Dockerfile + docker-compose (API + PostgreSQL), GitHub Actions CI (lint, test, build) |

\* Class labels are derived from the engine's overall score (weak supervision) on the 105-sample dataset, so this
measures how well the classifier reproduces the scoring policy, not agreement with human judges.

## Suggested resume bullets

- Built a production-style FastAPI platform that evaluates LLM answers in 5 languages with a hybrid rule-based + Gemini
  engine (relevance, correctness, clarity, safety, hallucination, duplication) and stores every evaluation in SQL.
- Designed a layered architecture (routers, services, repositories) with batch CSV evaluation, analytics APIs,
  PDF/Excel/CSV reporting, API-key auth, rate limiting, caching and connection pooling.
- Trained a Random Forest quality classifier and delivered 217 automated tests, Docker/Compose packaging and a
  GitHub Actions pipeline (lint, test, image build).
