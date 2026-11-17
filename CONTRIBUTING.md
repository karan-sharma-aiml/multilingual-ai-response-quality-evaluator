# Contributing

1. Fork the repository and create a branch: `git checkout -b feature/my-change`.
2. Install dependencies: `pip install -r requirements-dev.txt`.
3. Make your change with tests. Keep modules small, typed and free of hard-coded values.
4. Run the checks locally (the same ones CI runs):
   ```bash
   ruff check .
   pytest tests
   ```
5. Open a pull request describing the problem, the approach and any trade-offs.

Guidelines: follow the existing architecture (routes -> services -> repositories), add configuration through
`Settings` rather than constants scattered in code, never commit secrets, and do not add network calls to tests
(mock Gemini with `httpx.MockTransport` as in `tests/test_gemini_service.py`).
