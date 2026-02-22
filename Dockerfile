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

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import json; from framework.verify import run_verification; r=run_verification(); open('/app/logs/healthcheck.log','a').write(json.dumps(r['checks'])+'\n'); exit(0 if all(r['checks'].values()) else 1)" || exit 1

CMD ["scripts/validate.sh"]

FROM base AS streamlit

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

# ---------------------------------------------------------------------------
# All-in-one stage: single container with supervisord orchestrating all services
# ---------------------------------------------------------------------------
FROM base AS allinone

RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    wget \
    adduser \
    libfontconfig1 \
    musl \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://github.com/prometheus/prometheus/releases/download/v2.51.0/prometheus-2.51.0.linux-amd64.tar.gz \
    | tar xz --strip-components=1 -C /usr/local/bin/ \
      prometheus-2.51.0.linux-amd64/prometheus \
      prometheus-2.51.0.linux-amd64/promtool

RUN wget -qO /tmp/grafana.deb https://dl.grafana.com/oss/release/grafana_10.4.1_amd64.deb \
    && dpkg -i /tmp/grafana.deb \
    && rm /tmp/grafana.deb

COPY infra/prometheus-standalone.yml /etc/prometheus/prometheus.yml
COPY infra/alert_rules.yml /etc/prometheus/alert_rules.yml

COPY infra/grafana-standalone/provisioning /etc/grafana/provisioning
COPY infra/grafana/dashboards /var/lib/grafana/dashboards

ENV GF_SECURITY_ADMIN_PASSWORD=admin \
    GF_AUTH_ANONYMOUS_ENABLED=true \
    GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
    GF_PATHS_PROVISIONING=/etc/grafana/provisioning

COPY infra/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

COPY scripts/run_all_pipeline.sh /app/scripts/run_all_pipeline.sh
RUN chmod +x /app/scripts/run_all_pipeline.sh

RUN mkdir -p /app/results /app/logs /tmp/prometheus

EXPOSE 8501 3000 9090 9091

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
