# marl-forecast-game

Production-hardening iteration of a multi-agent adversarial forecasting game with:

- immutable simulation state
- four core agents (forecasting, adversary, defender, refactoring)
- pluggable runtime, disturbance, defense, and LLM refactor modules
- DRD-aligned source adapters (FRED / IMF / Polymarket) with resilient API fallbacks
- hybrid (real + synthetic) dataset support, poisoning checks, and chronological split enforcement
- structured observability via structlog + Prometheus counters/histograms
- reproducible verification report generation and extended property-based tests
- Haskell migration scaffold (Cabal + QuickCheck property)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python scripts/run_verification.py
python scripts/run_test_suite.py
```

## Docker Validation

### Fast path

```bash
docker build -t marl-forecast-game:test .
docker run --rm marl-forecast-game:test
```

The image includes a healthcheck that runs core verification checks.

### Manual containerized test steps (full plan)

1. Build the test image.
   ```bash
   docker build -t marl-forecast-game:test .
   ```
2. Run the default validation entrypoint (`scripts/validate.sh`).
   ```bash
   docker run --rm marl-forecast-game:test
   ```
3. Run the extended suite and aggregate report generation inside the container.
   ```bash
   docker run --rm -v "$PWD/planning:/app/planning" marl-forecast-game:test python scripts/run_test_suite.py
   ```
4. (Optional) Enable Ollama-backed checks from containerized tests.
   - Ensure Ollama is running on the host and reachable at `http://host.docker.internal:11434`.
   - Run:
     ```bash
     docker run --rm \
       --add-host host.docker.internal:host-gateway \
       -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
       marl-forecast-game:test \
       python -c "from framework.llm.ollama_interface import OllamaInterface; print(OllamaInterface(base_url='http://host.docker.internal:11434').is_available())"
     ```

## Data Ethics and Compliance Notes

- The framework only reads public APIs/sources listed in `DRD.md`.
- Source records retain provenance (`source`, `fetched_at`) for auditability.
- Ingestion includes outlier-based poisoning screening (`detect_poisoning_rows`).
- Chronological splitting is enforced to reduce leakage risk.
- API clients fail closed to synthetic proxies when endpoints are unreachable or credentials are unavailable.

## Haskell Migration Track

A parallel scaffold exists in `haskell/`:

- `src/Types.hs`: immutable state ADT equivalent.
- `src/Game.hs`: pure transition function.
- `test/Main.hs`: QuickCheck property (`maxSuccess = 1000`).

This track enables side-by-side parity checks while Python remains the primary runtime.
