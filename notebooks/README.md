# Notebooks

The notebook folder is intentionally small and reproducible.

- `quickstart_api.ipynb` shows the public `marl_forecast_game` import path.
- Use the sample-data path first so the notebook runs without external credentials.
- For real-data runs, set `FRED_API_KEY` before executing training or backtesting cells.

Recommended flow:

```bash
pip install -e ".[distributed,dev]"
jupyter notebook notebooks/quickstart_api.ipynb
```
