FROM haskell:9.6-slim AS haskell-build

WORKDIR /hs
COPY haskell/ .
RUN cabal update \
 && cabal build all \
 && cabal test all \
 && cp "$(cabal list-bin marl-forecast-game)" /usr/local/bin/marl-forecast-game

FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=WARNING

WORKDIR /app

COPY --from=haskell-build /usr/local/bin/marl-forecast-game /usr/local/bin/marl-forecast-game

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x scripts/validate.sh
RUN mkdir -p /app/logs

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import json; from framework.verify import run_verification; r=run_verification(); open('/app/logs/healthcheck.log','a').write(json.dumps(r['checks'])+'\n'); exit(0 if all(r['checks'].values()) else 1)" || exit 1

CMD ["scripts/validate.sh"]
