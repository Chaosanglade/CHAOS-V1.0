# CLAUDE.md — CHAOS V2 Project Context

## Project Overview

**CHAOS V2** is an institutional-grade quantitative trading system being built from scratch using 20 years of clean DukasCopy tick data. It is a ground-up rebuild that replaces earlier iterations (MultiPairScalperV3, CHAOS V1.0) which were derailed by silent data corruption bugs.

**Primary user:** Greg Anglade — 15-year quantitative trader, expert level, no hand-holding required. Treat as an expert peer.

**GitHub repo:** `Chaosanglade/chaos_v2` (private)

**Legacy repos (ARCHIVED, DO NOT EDIT):**
- `Chaosanglade/MultiPairScalperV3` — contains debugging history and the original 91%-accuracy models, kept for reference only

---

## CRITICAL: Canonical Paths — READ BEFORE TOUCHING ANY FILE

There has been persistent confusion about where the "real" project lives. The following are the ONLY correct paths. Any file access outside these paths is WRONG.

### Local machine (Greg's laptop)
- **Canonical code location:** `G:\My Drive\chaos_v1.0\`  
  (The directory is named `chaos_v1.0` for legacy reasons but this IS the CHAOS V2 project. Do not create a new `chaos_v2` folder.)
- **Raw tick data:** `C:\chaos_data\raw_ticks\{PAIR}\{PAIR}_full.csv`  
  (20 years of DukasCopy ticks, 5-column format: Time,Bid,Ask,BidVolume,AskVolume)
- **Working data:** `C:\chaos_data\cleaned_ticks\`, `C:\chaos_data\bars\`, `C:\chaos_data\features\`, `C:\chaos_data\validation\`, `C:\chaos_data\logs\`

### VPS (production deployment target)
- **Deployment tree:** `C:\CHAOS_PRODUCTION\`
- **User:** Administrator
- **Python venv:** `C:\CHAOS_PRODUCTION\.venv\`
- **IB Gateway:** port 7497 (paper), account DUP509332, clientId=2
- **NEVER connect to port 7496 (live trading)**

### FORBIDDEN paths — never use these:
- `C:\Users\Greg\Projects\MultiPairScalperV3\` — old repo location, stale
- `C:\Users\Administrator\Projects\MultiPairScalperV3\` — old VPS deployment
- `G:\My Drive\chaos_v2\` — does not exist, do not create

If you find yourself about to write to any forbidden path, STOP and ask Greg to confirm.

---

## CRITICAL: Verification Protocol

You have a documented history across previous sessions of fabricating success reports for actions that did not actually occur — claiming files were created when they were not, inventing commit hashes, claiming pushes that never happened. This has wasted hours of debugging time.

### Before reporting ANY task complete, you MUST:

1. **After creating a file:** Run `Get-ChildItem <path>` and paste the actual output confirming the file exists with non-zero size.
2. **After editing a file:** Run `Get-Content <path>` and confirm your changes are present.
3. **After git commit:** Run `git log -1 --stat` and paste the actual output. Never invent commit hashes.
4. **After git push:** Run `git status` and confirm "Your branch is up to date with 'origin/main'".
5. **Never recycle previous verification output.** If the same commit hash appears in two different task reports, you skipped verifying the second task.
6. **Never say "all verified" without pasting the actual command output.** The verification IS the output, not your summary of it.

---

## CRITICAL: Data Validation Protocol

This project exists because two prior iterations were derailed by silent data corruption bugs. Pipelines produced files that looked right (correct shape, correct columns, correct sizes) but contained corrupted values in specific columns — e.g., Open column with only 49 unique values across 1.12M rows. The corruption was invisible until bar-by-bar inspection weeks later.

**NEVER trust a data file without validating it first.**

### Before using ANY parquet/CSV/OHLCV file as input, you MUST:

1. **Print shape, columns, dtypes, date range.**

2. **Validate OHLCV integrity:**
   - Open, High, Low, Close should each have tens of thousands of unique values for multi-year M5 data. If any column has <10,000 unique values across 1M+ rows, it is CORRUPTED — stop and report.
   - High >= max(Open, Close) on every bar
   - Low <= min(Open, Close) on every bar
   - High >= Low on every bar
   - Bar-to-bar close delta should have realistic std (1-10 pips for major FX)
   - No gaps beyond weekends

3. **Sample and eyeball:** Print 5 random rows from the middle, first 5, last 5. Confirm values look like real market prices.

4. **Validate the OUTPUT of any data pipeline you write.** The LAST thing a tick aggregator, resampler, or feature generator does before declaring success must be validating its own output. Do not say "done" until validation passes.

### Clean data ground truth:

- **Raw source:** `C:\chaos_data\raw_ticks\{PAIR}\{PAIR}_full.csv`
- **Format:** 5 columns — Time, Bid, Ask, BidVolume, AskVolume
- **Precision:** millisecond timestamps
- **Date range:** 2005 – 2026 (20 years)
- **Pairs:** EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD, EURJPY, GBPJPY
- **Volume semantics:** BidVolume and AskVolume are DukasCopy ECN aggregated quote sizes in millions of units

### Known corruption to avoid:

Earlier `ohlcv_data/{PAIR}/{PAIR}_M5.parquet` files had only 49 unique Open values across 1.12M rows. These were all deleted in April 2026. If you find such files on disk, STOP and report.

---

## Project Phase Structure (A through K)

This is an 11-phase build. Each phase has a spec, acceptance criteria, and a quality gate before moving to the next. Greg is the reviewer and quality gate.

- **Phase A:** Tick aggregator — ticks to OHLCV at M1/M5/M15/M30/H1/H4/D1 with full microstructure (CURRENT PHASE)
- **Phase B:** Feature engineering — 400+ features on top of aggregated bars
- **Phase C:** Target generation and walk-forward splits
- **Phase D:** Model training — 21-brain ensemble
- **Phase E:** Signal validation and Monte Carlo
- **Phase F:** Ensemble voting and meta-labeling
- **Phase G:** Schema contract and ONNX export
- **Phase H:** ZeroMQ bridge and paper trading harness
- **Phase I:** Realistic backtest with costs
- **Phase J:** Risk management and drawdown gates
- **Phase K:** VPS deployment and live paper trading

Do not skip phases. Do not start Phase B until Phase A has passed all acceptance criteria.

---

## Environment Notes

### PowerShell quirks (local machine)
- `&&` does NOT work as a statement separator. Use newlines or `;`.
- Use `&` to invoke executables with spaces in path.
- Use `Get-ChildItem` not `ls`.

### Google Drive quirks
- `G:\My Drive\` is the mounted Drive path on Windows
- Writing large files to Drive is slow — use `C:\chaos_data\` for working data, only sync finished artifacts to Drive
- Drive may show "already up to date" even when content differs — verify with file size

### Python environment
- Local: Miniconda base env (pip may be broken — use conda install when possible)
- VPS: Python 3.13.12 in `C:\CHAOS_PRODUCTION\.venv`

---

## When in doubt

- Read the file before editing it
- Run the command and paste the real output
- Validate data before trusting it
- If something fails, say it failed
- Never say "done" without evidence
- Greg has 15 years of quant trading experience — treat him as an expert peer
- This project has been derailed by silent corruption twice. Be paranoid about data.
- The canonical path is `G:\My Drive\chaos_v1.0\`. Everything else is wrong.
