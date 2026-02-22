### Enhancement Specification for Qualitative State Space Expansion

#### 1. Dimensionality Reduction for Tabular Agents
Extend `ForecastState` dataclass with a new immutable field `qualitative_state: tuple[int, ...]` representing a bounded discrete tensor derived from qualitative narratives. The mapping function \( f: \text{Text} \to [-1, 0, 1]^k \) (where \( k \) is the number of extracted features, e.g., \( k=3 \) for sentiment, uncertainty, and forward guidance) is implemented as a deterministic extractor using `OllamaClient` with fixed seed \( s=42 \) and temperature \( T=0 \), enforcing JSON schema output `{"sentiment": int, "uncertainty": int, "guidance": int}`.

- **Mapping Efficacy**: Discrete mapping is selected over continuous embeddings to bound state space cardinality for Tabular Q-learning and WoLF-PHC convergence, as validated by arXiv:2508.02366 (hybrid LLM-RL with discrete signals) and arXiv:2510.10526 (FinGPT sentiment extraction). Continuous embeddings would induce state explosion, violating determinism in Q-table updates.
  
- **Integration Blueprint**:
  - In `AgentRegistry`, register tabular agents (e.g., `tabular_q: TabularQLearningAgent`, `wolf_phc: WoLFPHCAgent`) with extended observation space \( o_t = (quant_state_t, qualitative_state_t) \), where Q-updates are \( Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)] \) conditioned on discrete \( qualitative_state_t \).
  - In `SimulationConfig`, add parameter `qualitative_extractor_prompt: str = "Extract discrete [-1,0,1] for sentiment, uncertainty, guidance from {text} as JSON."` and `feature_dim: int = 3`.
  - Processing: During state transition, invoke `ollama_client.generate(text=beige_book_release, prompt=qualitative_extractor_prompt, seed=42, temp=0)` to yield reproducible tensor, appended immutably via `dataclasses.replace(forecast_state, qualitative_state=new_tensor)`.

#### 2. Temporal Alignment and State Decay
Introduce temporal decay to `qualitative_state` within `ForecastState` as a decayed tensor \( q_t = q_{release} \cdot e^{-\lambda (t - t_{release})} \), where \( q_{release} \) is the initial extracted tensor at release timestamp \( t_{release} \), \( \lambda > 0 \) is the decay rate (e.g., \( \lambda = 0.01 \) per timestep), and \( t \) is the current simulation step. Discretize post-decay via rounding \( \lfloor q_t \rceil \) to maintain bounded integers, ensuring persistence until next release but with exponential influence reduction.

- **Alignment Mechanism**: Align asynchronous releases (e.g., Beige Book at fixed dates) with uniform game timesteps by injecting only at matching \( t \) via chronological splitting; pre-release states use decayed prior \( q_{t-1} \) or zero-vector default. This enforces look-ahead bias prevention, as per arXiv:2510.05533 (temporal leakage mitigation).
  
- **Integration Blueprint**:
  - Extend `ForecastState` with fields `last_qual_release_ts: datetime`, `raw_qual_state: tuple[float, ...]`, and compute decayed `qualitative_state: tuple[int, ...]` on access via property `@property def qualitative_state(self): return tuple(round(v * exp(-self.decay_rate * (self.current_ts - self.last_qual_release_ts).days)) for v in self.raw_qual_state)`.
  - In `SimulationConfig`, add `decay_rate: float = 0.01` and `release_calendar: dict[datetime, str] = {dt: text_path for dt, text_path in historical_releases}` for timestamp-governed injection.
  - Transition: In game loop, if `current_ts in release_calendar`, load text, extract \( q_{release} \), update via `dataclasses.replace(state, raw_qual_state=q_release, last_qual_release_ts=current_ts)`; else decay existing.

#### 3. LLM-Native Strategy Runtimes
For prompt/LLM-backed agents, concatenate raw qualitative transcripts into context window as modular prompts following MAP4TS (arXiv:2510.23090): structure as `global_prompt = "Macro context: {macro_context_dict} | Qualitative: {beige_book_text} | PMI: {pmi_commentary}"` + `temporal_prompt = "Forecast CPI at t+1 given decayed qualitative influence e^{-λ Δt}"`. Merge multi-source texts via truncation to fit token limits, prioritizing recent releases.

- **Token Management**: Enforce max_tokens \( m = 4096 \) by summarizing older texts with Ollama (seed=42, T=0) if concatenation exceeds \( 0.8m \), using prompt "Summarize {old_text} to 200 tokens retaining sentiment/uncertainty."

- **Integration Blueprint**:
  - In `AgentRegistry`, register LLM agents (e.g., `prompt_llm: PromptLLMAgent`, `ollama_backed: OllamaAgent`) with runtime `def act(self, state): return ollama.generate(global_prompt.format(macro_context=state.macro_context, beige_book=state.qual_text if state.current_ts >= release_ts else '', pmi=... ), temp=0, seed=42)`.
  - In `SimulationConfig`, add `max_context_tokens: int = 4096`, `summary_prompt: str = "Summarize to 200 tokens..."`, and `modular_prompt_template: str` embedding MAP4TS schema.
  - State Extension: Add `qual_text: str` to `ForecastState` (immutable, updated only at releases), injected into macro_context dict as `macro_context['qual_narrative'] = qual_text`.

#### 4. Regime Detection Integration
Fuse qualitative narratives with quantitative data to compute a singular hidden `economic_regime: int` Markov state via a deterministic classifier: \( regime_t = \arg\max_i p(i | quant_features_t, decayed_qual_state_t) \), using offline Ollama with prompt "Classify regime [recession=0, expansion=1, stagflation=2] from {quant_dict} and {qual_tensor} as int." This fusion treats qualitative as conditioning variables rather than independent, enhancing regime transition matrix \( P(regime_{t+1} | regime_t, action_t) \) in MARL.

- **Fusion Efficacy**: Integrated fusion improves multi-agent consensus (arXiv:2410.14383, MARLIN negotiation), avoiding state fragmentation.

- **Integration Blueprint**:
  - Extend `ForecastState` with `economic_regime: int`, computed immutably in transition via `regime = ollama.generate(classify_prompt.format(quant=state.quant_dict, qual=state.qualitative_state), seed=42, temp=0).parse_int()`.
  - In `AgentRegistry`, condition all agents' policies on `regime_t` (e.g., tabular Q-tables keyed by `(quant, regime)`, LLM prompts appending "Regime: {regime}").
  - In `SimulationConfig`, add `regime_classes: int = 3`, `regime_prompt: str = "Classify regime..."`, enabling hierarchical topologies where ensemble aggregators negotiate regime via shared LLM calls before Q-updates.

### Specification for Qualitative Test Data Ingestion

#### 1. Data Source Identification and Adapters Extension
Extend `framework/data_sources/base.py` with new `QualitativeAdapter` protocol inheriting `SourceAdapter`, defining methods `fetch_releases(start_dt: datetime, end_dt: datetime) -> list[NormalizedQualRecord]` where `NormalizedQualRecord = NamedTuple(timestamp: datetime, source_id: str, text: str, metadata: dict[str, Any])`. Adapters fetch timestamped text without API keys, using public HTTP downloads or scraping where legal, cached via `checksum_integrity_check` with SHA-256 hashing for reproducibility.

- **Beige Book Adapter**: Source from Federal Reserve public archives (https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook-archive.htm). Fetch HTML/PDF for 8 annual releases (biweekly schedule approximation: mid-Jan, early-Mar, mid-Apr, early-Jun, mid-Jul, early-Sep, mid-Oct, early-Dec). Convert PDF to text via `pypdf` (add to requirements.txt if absent). Test dataset: 2010-2025 releases, e.g., 2023-01-18 (timestamp: 2023-01-18T14:00:00 EST, text: full narrative extract).
  
- **PMI Commentary Adapter**: Source from ISM public reports (https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/pmi/). Monthly releases (first business day). Fetch HTML, extract commentary section via XPath `//div[@class='report-commentary']`. Test dataset: Manufacturing PMI 2015-2025, e.g., 2023-02-01 (timestamp: 2023-02-01T10:00:00 EST, text: anecdotal comments fusion).
  
- **Executive Earnings Transcripts Adapter**: Source from Seeking Alpha public transcripts (https://seekingalpha.com/earnings/earnings-call-transcripts), filtered to S&P 500 tickers (e.g., AAPL, MSFT). Quarterly releases aligned to fiscal calendars. Fetch via URL pattern `https://seekingalpha.com/article/{id}-ticker-earnings-call-transcript`, parse text body. Test dataset: 50 transcripts 2020-2025, e.g., AAPL Q1 2023 (timestamp: 2023-02-02T17:00:00 EST, text: CEO/CFO remarks).

- **Integration Blueprint**: In `SimulationConfig`, add `qual_adapters: list[str] = ['beige_book', 'pmi', 'earnings']` and `test_data_dir: Path = Path('data/test_qualitative/')` for static downloads. During ingestion, merge into chronological queue `qual_queue: deque[NormalizedQualRecord]`, popping at matching `current_ts >= record.timestamp` to enforce look-ahead prevention.

#### 2. Chronological Splitting and Test Dataset Construction
Construct test dataset via strict chronological partitioning: train (2010-01-01 to 2020-12-31), valid (2021-01-01 to 2023-12-31), test (2024-01-01 to 2025-12-31). Fetch public data into CSV manifest `test_manifest.csv` with columns [source_id, timestamp, url, local_path, checksum]. Total records: ~200 (80 Beige Book, 120 PMI, variable earnings capped at 50). Ensure determinism via fixed seed `s=42` for any randomization (e.g., earnings subsample).

- **Quantitative Alignment**: Pair with existing FRED series (CPIAUCSL monthly at mid-month, e.g., 2023-02-14T08:30:00 EST) and Polymarket (continuous, historical via public API https://api.polymarket.com/markets?slug=cpi-inflation-december-2023, archived via Wayback Machine if needed). Test fusion: For each qual release, align to nearest quant timestep \( t_{align} = \arg\min_{t \in quant_ts} |t - qual_ts| \), injecting only if \( t_{align} \geq qual_ts \).
  
- **Integration Blueprint**: Extend `framework/data.py` with `def build_qual_dataset(config: SimulationConfig) -> dict[datetime, str]:` returning timestamp-text map. In game init, load via `qual_data = build_qual_dataset(config)`, integrated into `ForecastState` at transitions matching \( state.current_ts \).

#### 3. Verification and Reproducibility Checks
Implement 100-run verification for qualitative ingestion: Hash text inputs pre-extraction, assert identical `qualitative_state` tensors across runs given seed=42. Poisoning detection via text length z-score \( z = (len(text) - \mu)/\sigma > 3 \), where \(\mu, \sigma\) computed over test dataset. Public data freshness: Script `scripts/fetch_test_data.py` with HTTP ETag checks, raising if modified.

- **Test Scenarios**: Add to `scripts/run_validation_scenarios.py`: Scenario 'qual_ingestion' (10k rounds with async releases), 'decay_verification' (assert \( q_t = q_0 e^{-\lambda (t-t_0)} \) numerically for \(\lambda=0.01\), tolerance 1e-6), 'bias_prevention' (assert no future text in past states via timestamp assertions).
  
- **Integration Blueprint**: In `AgentRegistry`, condition verifications on qual-enabled agents; extend `SimulationConfig` with `enable_qual: bool = True`, `verification_runs: int = 100`.

#### 4. Public Fetching Pipeline
Define pipeline in `scripts/fetch_test_data.py`: For each adapter, iterate historical dates, download to `data/test_qualitative/{source_id}/{yyyy-mm-dd}.txt`, compute checksum. No API keys required; use `requests` for HTTP (add to requirements.txt if absent). Total size bound: <100MB text.

- **Earnings Subset**: Fixed tickers ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'] for reproducibility, fetching ~10/quarter.
  
- **Integration Blueprint**: Hook into existing caching: `if not local_path.exists() or not checksum_match(): download_and_cache(url)`. Ensure all test data URLs are public and non-paywalled.

This specification defines the rigorous architectural extension of the `marl-forecast-game` framework to incorporate high-dimensional qualitative narratives into a pure-functional, deterministic Markov game.

The objective is to transform unstructured macroeconomic sentiment into a bounded, discrete state space suitable for Multi-Agent Reinforcement Learning (MARL) while maintaining zero-leakage chronological integrity and 100-run deterministic reproducibility.

---

## 1. Mathematical Foundation and Invariants

The system operates as a Discrete-Time Markov Game. We extend the state space  to include a qualitative manifold .

### 1.1 State Space Extension

The augmented state at time  is defined as the tuple:



Where:

* : Quantitative features (FRED, IMF).
* : The **Decayed Qualitative Tensor**, representing  discrete sentiment features.
* : The **Economic Regime**, a latent variable inferred via LLM fusion.
* : The timestamp and metadata for lineage tracking.

### 1.2 The Qualitative Transition Invariant

For any state transition , the following property must hold:



This enforces a strict "no-look-ahead" invariant. Qualitative data is injected only at .

---

## 2. Qualitative State Space Expansion

### 2.1 Dimensionality Reduction for Tabular Agents

To prevent the curse of dimensionality in Tabular Q-learning and WoLF-PHC, raw text is mapped to a discrete tensor.

* **Extraction Function ():** Uses `OllamaClient` with a system prompt enforcing a strict JSON schema.
* **Parameters:** `model="llama3"`, `seed=42`, `temperature=0`, `top_k=1`.
* **Schema:** `{"sentiment": int, "uncertainty": int, "guidance": int}` where values .
* **Invariant:** The same input text must yield the same integer triplet across  of trials.

### 2.2 Temporal Decay Mechanics

Qualitative signals exhibit information entropy over time. We model this as exponential decay:



Where:

*  is the `decay_rate` defined in `SimulationConfig`.
*  denotes the nearest integer rounding to maintain the discrete state space .

### 2.3 Regime Detection Fusion

The system shall implement a deterministic regime classifier that fuses quantitative and qualitative signals:



The `economic_regime` becomes a primary key for the Q-table in `TabularQLearningAgent`, effectively partitioning the agent's policy by macro context.

---

## 3. Data Ingestion & Lineage Specification

The framework shall implement three new specialized adapters inheriting from the `SourceAdapter` protocol.

### 3.1 Qualitative Source Matrix

| Source | Frequency | Feature Extraction Focus | Alignment Strategy |
| --- | --- | --- | --- |
| **Beige Book** | 8x / Year | Regional sentiment, labor tightness |  Publication Date |
| **PMI Commentary** | Monthly | Supply chain friction, price pressure | First business day |
| **Earnings Transcripts** | Quarterly | Guidance, CapEx intentions | Post-market release |

### 3.2 Ingestion Pipeline Invariants

1. **Checksum Integrity:** All downloaded `.txt` or `.pdf` artifacts must be hashed via SHA-256. If `local_hash != remote_hash`, the simulation must abort to prevent non-deterministic training.
2. **Poisoning Detection:** A `z-score` check on text length is mandatory. Records with  are flagged as "poisoned" (e.g., truncated downloads or empty responses) and excluded.

---

## 4. Implementation Requirements

### 4.1 `ForecastState` Update (framework/types.py)

The `ForecastState` frozen dataclass must be extended:

```python
@dataclass(frozen=True)
class ForecastState:
    # ... existing fields ...
    qualitative_state: tuple[int, ...]     # The discrete tensor q_t
    raw_qual_text: Optional[str]           # The raw narrative (for LLM agents)
    economic_regime: int                   # The fused regime index
    last_qual_update: datetime             # For decay calculation

```

### 4.2 Tabular Agent Observation Space

The `TabularQLearningAgent` must update its state-keying logic:

```python
def _get_state_key(self, state: ForecastState) -> tuple:
    # Keying by discrete bins + qualitative tensor + regime
    return (self._bin_quant(state.current_value), state.qualitative_state, state.economic_regime)

```

---

## 5. Execution Sequence (SDD-Strict)

Tasks must be executed in the following topological order. No iterative phasing is permitted.

1. **Core Types:** Extend `ForecastState` and `SimulationConfig` with qualitative fields.
2. **Adapter Layer:** Implement `BeigeBookAdapter`, `PMIAdapter`, and `EarningsAdapter` in `framework/data_sources/`.
3. **Deterministic Extraction:** Build the `QualitativeExtractor` utility using `OllamaClient` with fixed seeds.
4. **Decay Logic:** Implement the `@property` for exponential decay within `ForecastState`.
5. **Regime Fusion:** Implement the `RegimeClassifier` and hook it into the `evolveState` transition in `game.py`.
6. **Agent Updates:** Modify `AgentRegistry` to pass augmented states to Tabular and LLM agents.
7. **Validation:** Deploy the 3 new validation scenarios: `qual_ingestion`, `decay_verification`, and `regime_consistency`.

---

## 6. Falsification & Definition of Done

The enhancement is considered complete only when the following empirical thresholds are met:

* **Determinism:** `scripts/run_verification.py` confirms  identical outputs for a 1000-round simulation using qualitative data.
* **Decay Accuracy:** A Hypothesis property test confirms that , the magnitude of  is monotonically non-increasing.
* **Look-ahead Prevention:** A check asserts that the LLM context window never contains text with a timestamp .
* **Convergence:** `TabularQLearningAgent` must show a non-zero win-rate improvement (relative to quantitative-only baseline) in the "Volatility Burst" scenario within 500 training episodes.
