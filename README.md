# Multilingual AI Response Quality Evaluator

[![CI](https://github.com/<your-username>/multilingual-ai-response-evaluator/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/multilingual-ai-response-evaluator/actions)
![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)
![License](https://img.shields.io/badge/license-MIT-green)

An evaluation platform for Large Language Model (LLM) answers written in **English, Hindi, French, Spanish and German**.
It scores every response for relevance, correctness, clarity, safety, hallucination risk and duplication, optionally
blends in a Gemini judgement, stores each evaluation in SQL, and exposes analytics, reports and a quality classifier.

## Table of contents

1. [Project overview](#project-overview) - 2. [Features](#features) - 3. [Architecture](#architecture) -
4. [Folder structure](#folder-structure) - 5. [Installation](#installation) - 6. [Environment variables](#environment-variables) -
7. [Database setup](#database-setup) - 8. [Gemini configuration](#gemini-configuration) - 9. [Running the APIs](#running-the-apis) -
10. [API reference](#api-reference) - 11. [Analytics APIs](#analytics-apis) - 12. [Reports](#reports) - 13. [ML pipeline](#ml-pipeline) -
14. [Visual analytics](#visual-analytics) - 15. [Security](#security) - 16. [Performance](#performance) - 17. [Docker](#docker) -
18. [Testing and CI](#testing-and-ci) - 19. [Deployment](#deployment) - 20. [Future improvements](#future-improvements) -
21. [Contributing](#contributing) - 22. [License](#license)

## Project overview

| Input | Output |
|-------|--------|
| `question`, `language`, `response`, optional `reference_answer` | relevance, correctness, clarity, safety, hallucination (risk), duplicate %, overall score, warnings |

Part 1 is a deterministic, explainable rule-based engine (TF-IDF, lexicons, heuristics). Part 2 adds the REST API, Gemini
integration, SQL persistence and batch evaluation. Part 3 adds analytics, reporting, ML, security, performance work,
Docker and CI/CD.

## Features

* **Hybrid evaluation** - rule engine + Gemini, weighted merge, automatic fallback when Gemini is unavailable
* **5 languages**, 105-record multilingual sample dataset (good, incorrect, hallucinated, repeated, unsafe, short, long)
* **Batch CSV evaluation** with per-row error reporting and concurrent Gemini calls
* **Compare endpoint** to pick the better of two responses with a reason
* **Analytics APIs**: overview, languages, categories, hallucination types, low-quality responses, daily/weekly/monthly trends
* **Reports**: CSV, Excel (formatted + summary sheet), PDF
* **ML**: Random Forest quality classifier (Excellent / Good / Average / Poor) with full metrics
* **Visual analytics**: 6 Matplotlib charts incl. correlation heatmap
* **Security**: API keys, rate limiting, strict CORS, security headers, input sanitisation, CSV-injection protection
* **Performance**: TTL cache, composite indexes, connection pooling, async health check, concurrent Gemini calls, GZip
* **Ops**: Dockerfile, docker-compose (API + PostgreSQL), GitHub Actions (lint, test, build), 217 tests

## Architecture

## System architecture

```mermaid
flowchart LR
    Client["Client<br/>(Swagger UI, Postman, apps)"] -->|"HTTP + X-API-Key"| MW

    subgraph API["FastAPI application"]
        MW["Middleware<br/>CORS, GZip, request logging,<br/>security headers, rate limiting"]
        MW --> Routers["Routers<br/>health, evaluate, compare, batch,<br/>results, analytics, reports"]
        Routers --> ES["EvaluationService"]
        Routers --> BS["BatchEvaluationService"]
        Routers --> AS["AnalyticsService"]
        Routers --> RS["ReportService"]
        BS --> ES
        ES --> RULES["Rule-based engine<br/>relevance, correctness, clarity, safety,<br/>hallucination, duplicates, rules"]
        ES --> GS["GeminiEvaluationService<br/>retries, timeouts, JSON validation"]
        ES --> DS["DatabaseService"]
        RS --> DS
        AS --> CACHE[("TTL cache")]
        AS --> REPO
        DS --> REPO["Repositories"]
    end

    GS -->|"HTTPS"| GEMINI["Google Gemini API"]
    REPO --> DB[("SQLite / PostgreSQL")]

    subgraph OFFLINE["Offline tooling"]
        ML["ML pipeline<br/>Random Forest quality classifier"]
        CH["Chart generator<br/>Matplotlib"]
    end
    DB -.-> ML
    DB -.-> CH
```

## Evaluation flow

```mermaid
flowchart TD
    A["POST /evaluate"] --> B["Validate and sanitise input<br/>language, length, control characters"]
    B -->|"invalid"| E1["422 validation_error"]
    B --> C["Rule engine<br/>6 metrics + warnings"]
    C --> D{"use_gemini?"}
    D -->|"no"| H["Rule-based scores"]
    D -->|"yes"| F["Gemini evaluation<br/>relevance, correctness, clarity, safety"]
    F -->|"timeout, 429, 5xx, invalid JSON"| G["Retry with backoff"]
    G -->|"still failing"| H
    G -->|"ok"| F
    F -->|"ok"| M["Merge<br/>weighted blend, safety = stricter score"]
    M --> O["Overall score<br/>duplicate penalty, warnings, safety cap"]
    H --> O
    O --> S["Save to database<br/>response, result, audit log"]
    S --> X["Invalidate analytics cache"]
    X --> R["201 JSON response"]
```

## Database ER diagram

```mermaid
erDiagram
    LANGUAGES ||--o{ RESPONSES : "written in"
    RESPONSES ||--o{ EVALUATION_RESULTS : "scored by"
    EVALUATION_RESULTS ||--o{ EVALUATION_LOGS : "audited by"

    LANGUAGES {
        int id PK
        string code UK
        string name
    }
    RESPONSES {
        int id PK
        int language_id FK
        text question
        text response
        text reference_answer
        string category
        datetime created_at
    }
    EVALUATION_RESULTS {
        int id PK
        int response_id FK
        float relevance_score
        float correctness_score
        float clarity_score
        float safety_score
        float hallucination_score
        float duplicate_percentage
        float overall_score
        string evaluation_method
        json warnings
        json details
        datetime created_at
    }
    EVALUATION_LOGS {
        int id PK
        int evaluation_result_id FK
        string level
        string event
        text message
        float latency_ms
        datetime created_at
    }
```

`evaluation_logs.evaluation_result_id` is nullable: audit rows survive the deletion of their result.

See [docs/architecture.md](docs/architecture.md). Layering: `routes -> services -> repositories -> database`; evaluators are
independent, injectable classes behind the `BaseEvaluator` interface.

## Folder structure

```text
multilingual-ai-response-evaluator/
├── .github/
│   └── workflows/
│       └── ci.yml
├── backend/
│   └── app/
│       ├── analytics/
│       │   ├── __init__.py
│       │   ├── charts.py
│       │   ├── frames.py
│       │   ├── generate_charts.py
│       │   ├── hallucination_types.py
│       │   └── service.py
│       ├── api/
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── analytics.py
│       │   │   ├── batch.py
│       │   │   ├── evaluation.py
│       │   │   ├── health.py
│       │   │   ├── reports.py
│       │   │   └── results.py
│       │   ├── __init__.py
│       │   ├── dependencies.py
│       │   └── filters.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── cache.py
│       │   ├── config.py
│       │   ├── error_handlers.py
│       │   ├── exceptions.py
│       │   ├── logging.py
│       │   ├── middleware.py
│       │   ├── rate_limit.py
│       │   └── security.py
│       ├── database/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── init_db.py
│       │   └── session.py
│       ├── evaluator/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── clarity.py
│       │   ├── correctness.py
│       │   ├── duplicate_detector.py
│       │   ├── hallucination.py
│       │   ├── overall_score.py
│       │   ├── pipeline.py
│       │   ├── relevance.py
│       │   ├── rule_engine.py
│       │   └── safety.py
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── labels.py
│       │   ├── predict.py
│       │   └── train_model.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── entities.py
│       │   └── evaluation_record.py
│       ├── repositories/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── evaluation_log_repository.py
│       │   ├── evaluation_result_repository.py
│       │   ├── language_repository.py
│       │   └── response_repository.py
│       ├── schemas/
│       │   ├── __init__.py
│       │   ├── analytics.py
│       │   ├── api.py
│       │   └── evaluation.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── batch_service.py
│       │   ├── database_service.py
│       │   ├── evaluation_service.py
│       │   ├── gemini_service.py
│       │   ├── mappers.py
│       │   └── report_service.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── language.py
│       │   ├── lexicons.py
│       │   ├── paths.py
│       │   ├── phrase_matcher.py
│       │   ├── sanitize.py
│       │   ├── scoring.py
│       │   ├── similarity.py
│       │   └── text.py
│       ├── __init__.py
│       └── main.py
├── datasets/
│   ├── batch_sample.csv
│   ├── generate_dataset.py
│   └── sample_dataset.csv
├── docker/
│   └── entrypoint.sh
├── docs/
│   ├── screenshots/
│   │   └── README.md
│   ├── architecture.md
│   └── RESUME_METRICS.md
├── ml_artifacts/
│   └── quality_classifier.joblib
├── postman/
│   ├── Multilingual_AI_Evaluator.postman_collection.json
│   └── Multilingual_AI_Evaluator.postman_environment.json
├── reports/
│   ├── charts/
│   │   ├── category_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── hallucination_distribution.png
│   │   ├── language_distribution.png
│   │   ├── quality_score_distribution.png
│   │   └── safety_score_distribution.png
│   ├── ml/
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png
│   │   └── metrics.json
│   └── sample_results.csv
├── tests/
│   ├── conftest.py
│   ├── test_analytics.py
│   ├── test_api.py
│   ├── test_batch.py
│   ├── test_correctness_clarity.py
│   ├── test_database.py
│   ├── test_duplicate_detector.py
│   ├── test_gemini_service.py
│   ├── test_hallucination_detector.py
│   ├── test_ml.py
│   ├── test_overall_score.py
│   ├── test_performance.py
│   ├── test_pipeline_and_dataset.py
│   ├── test_relevance.py
│   ├── test_reports.py
│   ├── test_rule_engine.py
│   ├── test_safety.py
│   └── test_security.py
├── .dockerignore
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── pyproject.toml
├── pytest.ini
├── README.md
├── requirements-dev.txt
├── requirements.txt
└── run_demo.py
```

## Installation

```bash
git clone https://github.com/<your-username>/multilingual-ai-response-evaluator.git
cd multilingual-ai-response-evaluator
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # use requirements.txt for runtime only
cp .env.example .env
```

## Environment variables

Copy `.env.example` and adjust. Highlights (see the file for all options and comments):

| Variable | Default | Purpose |
|----------|---------|---------|
| `APP_ENV` | `development` | `development`, `test` or `production` (production forbids `CORS_ORIGINS=*`) |
| `DATABASE_URL` | `sqlite:///./evaluator.db` | SQLAlchemy URL (PostgreSQL supported) |
| `API_KEYS` | _empty_ | Comma-separated keys; empty disables authentication (development only) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` | `120` / `60` | Per client (API key or IP) |
| `GEMINI_API_KEY` | _empty_ | Enables Gemini; empty = rule-based only |
| `GEMINI_MODEL`, `GEMINI_WEIGHT` | `gemini-2.5-flash`, `0.6` | Model and its share in merged scores |
| `GEMINI_CONCURRENCY` | `4` | Parallel Gemini calls in batch/compare |
| `CACHE_TTL_SECONDS` | `60` | Analytics cache lifetime |
| `DB_POOL_SIZE`, `DB_MAX_OVERFLOW` | `10`, `20` | Pool for server databases |
| `ENABLE_DOCS` | `true` | Serve Swagger/ReDoc/OpenAPI |
| `LOW_QUALITY_THRESHOLD` | `50` | Used by analytics |

## Database setup

Tables, indexes and the five supported languages are created automatically on start-up (`AUTO_CREATE_TABLES=true`) or by
`python -m app.database.init_db` (run from `backend/`; used by the Docker entrypoint). The step is idempotent and also adds
indexes that are missing on older databases. For PostgreSQL:

```bash
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/evaluator
```

Schema: `languages`, `responses`, `evaluation_results`, `evaluation_logs` (ER diagram above). Schema migrations with Alembic are listed
under future improvements.

## Gemini configuration

1. Create a key in Google AI Studio and set `GEMINI_API_KEY` in `.env`.
2. Optionally tune `GEMINI_MODEL`, `GEMINI_WEIGHT` (0-1), `GEMINI_TIMEOUT_SECONDS`, `GEMINI_MAX_RETRIES`.
3. `GET /health` reports `gemini.status` (`ok` or `not_configured`).

Merging: relevance, correctness, clarity = `w * gemini + (1 - w) * rule`; safety = the stricter (lower) of both; hallucination and
duplicate % always come from the rule engine; the overall score is recomputed. Timeouts, HTTP 429 (honouring `Retry-After`), 5xx and
invalid JSON are retried with exponential backoff; if Gemini still fails the evaluation falls back to rule-based scores and the reason is
returned in `gemini.error`.

## Running the APIs

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Swagger UI: <http://localhost:8000/docs> - ReDoc: `/redoc`. When `API_KEYS` is set, send `X-API-Key: <key>` (use **Authorize** in Swagger).
Every response carries `X-Request-ID`, `X-Process-Time-Ms` and `X-RateLimit-*` headers. Errors share one shape:
`{"error": {"code": "...", "message": "...", "details": ...}}`.

![Swagger UI](docs/screenshots/swagger-ui.png)
![Evaluate response](docs/screenshots/evaluate-response.png)

> Screenshot placeholders: add your own captures (see `docs/screenshots/README.md`).

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | API, database and Gemini status (public) |
| POST | `/evaluate` | Evaluate one response, store it, return scores (201) |
| POST | `/compare` | Compare two responses to one question |
| POST | `/batch-evaluate` | CSV upload (`question,response,language`; optional `reference_answer,category,id`) |
| GET | `/results` | Pagination, sorting (`sort_by`, `order`), filters (`language`, `category`, `evaluation_method`, score range) |
| GET | `/results/{id}` | One evaluation |
| DELETE | `/results/{id}` | Delete an evaluation |
| GET | `/analytics/...` | See below |
| GET | `/reports/csv`, `/reports/excel`, `/reports/pdf` | Downloads |

Example:

```bash
curl -X POST http://localhost:8000/evaluate -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"question":"What is cloud computing?","response":"Cloud computing means renting servers over the internet.","language":"English","reference_answer":"Cloud computing delivers computing services over the internet."}'
```

Validation (HTTP 422): missing fields, unsupported language, empty response, maximum lengths (question 2,000; response and reference 20,000
characters). Batch: file must be `.csv` (400), at most `MAX_UPLOAD_BYTES` (413) and `MAX_BATCH_ROWS` rows; bad rows are listed in
`failed_records`. A Postman collection and environment are in `postman/`.

## Analytics APIs

| Endpoint | Returns |
|----------|---------|
| `GET /analytics/overview` | Total evaluations, average overall/relevance/correctness/clarity/safety/hallucination, low-quality count |
| `GET /analytics/languages` | Per-language totals and averages, best and worst language |
| `GET /analytics/categories` | Per-category statistics (uncategorised rows grouped as `Uncategorized`) |
| `GET /analytics/hallucinations` | Hallucination types (fake URLs, unsupported statistics/claims, contradictions, made-up entities, ungrounded content) with `frequency`, `percentage` (share of all detections) and `prevalence_percentage` (share of evaluations) |
| `GET /analytics/low-quality` | Evaluations with overall score `< threshold` (default 50), worst first, paginated, filterable |
| `GET /analytics/trends` | Daily, weekly (ISO week start) and monthly series; optional `period`, `start_date`, `end_date` |

Aggregates are computed in SQL and cached for `CACHE_TTL_SECONDS`; any write invalidates the cache. Evaluations stored before hallucination types
were recorded are re-analysed on demand.

## Reports

`GET /reports/csv|excel|pdf` accept the same filters as `/results` plus `start_date`, `end_date` and `limit`. Each row has question,
language, category, response, all scores, Gemini scores and reasoning, overall score, method, warnings and UTC timestamp. Text cells are
protected against CSV/Excel formula injection. CSV is UTF-8 with BOM (opens correctly in Excel); Excel adds a formatted header, filters, a colour scale and a
`Summary` sheet. **PDF note:** ReportLab does not shape Devanagari, so characters the PDF font cannot draw are shown as `?`; CSV and Excel
keep full Unicode. Set `REPORT_FONT_PATH` to use a different TrueType font.

## ML pipeline

```bash
cd backend
python -m app.ml.train_model                 # trains on the sample dataset
python -m app.ml.predict --relevance 90 --correctness 85 --clarity 80 --safety 100 --hallucination 0 --duplicate 0
```

* Features: relevance, correctness, clarity, safety, hallucination, duplicate %. Classes: Excellent (>=85), Good (>=70), Average (>=50), Poor (<50).
* Model: `RandomForestClassifier` (300 trees, balanced class weights); stratified 75/25 split plus stratified k-fold cross-validation.
* Outputs: accuracy, weighted/macro precision, recall and F1, confusion matrix, classification report, feature importances. Saved to `ml_artifacts/quality_classifier.joblib`
  (joblib) and `reports/ml/` (`metrics.json`, `classification_report.txt`, `confusion_matrix.png`).
* Sources: `--source dataset` (default), `--source csv --csv file.csv`, `--source db`.
* Reference run on the 105-sample dataset: **92.6% test accuracy, 95.2% cross-validated accuracy**.

**Important:** labels are derived from the engine's overall score, so the classifier measures how well the six metrics reproduce the scoring
policy. For an independent judge, train on human-labelled data. Only load model files you trained yourself (joblib can execute code).

## Visual analytics

```bash
cd backend
python -m app.analytics.generate_charts                    # sample dataset
python -m app.analytics.generate_charts --source db        # stored evaluations
```

Writes language distribution, quality score distribution, hallucination distribution, safety score distribution, category distribution and a
correlation heatmap to `reports/charts/`.

![Correlation heatmap](reports/charts/correlation_heatmap.png)
![Quality distribution](reports/charts/quality_score_distribution.png)

## Security

* **API key authentication** (`X-API-Key`, constant-time comparison) on every router except `/health`; disabled only when `API_KEYS` is empty.
* **Rate limiting**: sliding window per API key or IP, `429` with `Retry-After`. State is per process - use a shared store or gateway for multi-instance limits.
* **Secure CORS**: explicit origins, limited methods and headers; production refuses `*`.
* **Security headers**: `nosniff`, `X-Frame-Options`, `Referrer-Policy`, `Cache-Control: no-store`, HSTS in production.
* **Input sanitisation**: Unicode NFC, control/bidi/invisible characters removed (Hindi ZWJ/ZWNJ kept), length limits, extra fields rejected; exports escape formulas.
* **Configuration**: everything through environment variables; secrets are never logged; Docker runs as a non-root user with a read-only filesystem.
* Behind a reverse proxy set `TRUST_PROXY_HEADERS=true` so rate limiting sees client IPs; terminate TLS at the proxy.

## Performance

* TTL cache for analytics with write-through invalidation
* Composite indexes (`created_at+overall_score`, `evaluation_method+created_at`, `language_id+category`, `event+created_at`), created idempotently at start-up
* Connection pooling with pre-ping for PostgreSQL (`DB_POOL_*`); in-memory SQLite uses a static pool for tests
* Gemini calls run concurrently (`GEMINI_CONCURRENCY`) in `/compare` and batch evaluation; the health endpoint is `async`
* Database-bound and CPU-bound endpoints are plain `def` handlers, which FastAPI runs in a thread pool (SQLAlchemy sessions are synchronous)
* GZip responses, streamed aggregation queries

## Docker

```bash
cp .env.example .env     # set POSTGRES_PASSWORD, API_KEYS, CORS_ORIGINS (and GEMINI_API_KEY)
docker compose up --build -d
curl http://localhost:8000/health
```

The image is multi-stage, runs as a non-root user, has a health check and starts through `docker/entrypoint.sh` (creates tables once, then
starts uvicorn workers). Compose runs the API with PostgreSQL, a read-only root filesystem and dropped capabilities. SQLite-only run:

```bash
docker build -t multilingual-ai-evaluator .
docker run -p 8000:8000 -e CORS_ORIGINS=https://app.example.com -e API_KEYS=change-me -v evaluator-data:/app/data multilingual-ai-evaluator
```

## Testing and CI

```bash
pytest tests          # 217 tests, no network or API key needed
ruff check .
```

Coverage: engine, REST API, repositories, Gemini client (mocked HTTP transport), batch evaluation, analytics, reports, ML pipeline, charts, security, caching and
concurrency. GitHub Actions (`.github/workflows/ci.yml`) runs **lint -> test -> Docker build** on every push and pull request.

## Deployment

1. Provision PostgreSQL and set `DATABASE_URL`, `APP_ENV=production`, `API_KEYS`, explicit `CORS_ORIGINS`, optionally `GEMINI_API_KEY`.
2. Build and run the image (or `docker compose up -d`) behind a TLS-terminating reverse proxy (nginx, Caddy, a cloud load balancer).
3. Set `TRUST_PROXY_HEADERS=true` and `FORWARDED_ALLOW_IPS` to the proxy address; set `ENABLE_DOCS=false` if the API should not be self-documenting.
4. Monitor `GET /health` and `logs/evaluator.log`; back up the database.

## Future improvements

* Alembic migrations and a Redis-backed cache / rate limiter for multi-instance deployments
* Async database driver and a fully async Gemini client
* Human-labelled data for the ML model and a `/predict-quality` endpoint
* Devanagari-capable PDF rendering (text shaping), a web dashboard, role-based access and per-key quotas
* Additional languages and embedding-based semantic similarity

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md): fork, branch, add tests, run `ruff check .` and `pytest tests`, open a pull request.

## License

Released under the [MIT License](LICENSE).
#   m u l t i l i n g u a l - a i - r e s p o n s e - q u a l i t y - e v a l u a t o r  
 