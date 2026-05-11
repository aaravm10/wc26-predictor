# wc26-predictor
A data-driven 2026 FIFA World Cup prediction system using an ensemble of Elo ratings, XGBoost, and a Neural Network, combined with a Monte Carlo tournament simulator.

## Project Architecture
Full architecture breakdown in `docs/ARCHITECTURE.md`

## Pipeline Overview

1. **Data Collection** — historical match results, FIFA rankings, player stats, betting odds
2. **Feature Engineering** — Elo ratings, form, head to head, gut feeling features
3. **Models** — Elo baseline, XGBoost, Neural Network ensembled together
4. **Simulation** — 10,000 Monte Carlo tournament simulations
5. **Output** — win probabilities per team, most likely bracket

## Repo Structure
wc26-predictor/
├── data/
│   ├── raw/          ← downloaded datasets, gitignored
│   ├── processed/    ← cleaned and merged data
│   └── external/     ← betting odds, sentiment data
├── notebooks/        ← exploration and analysis
├── src/
│   ├── data/         ← data collection and loading
│   ├── features/     ← feature engineering
│   ├── models/       ← elo, xgboost, neural net, ensemble
│   ├── simulation/   ← tournament bracket simulator
│   └── evaluation/   ← validation against past world cups
├── tests/            ← unit tests for each module
├── docs/             ← architecture and technical docs
└── results/          ← simulation outputs, gitignored

## Setup

```bash
git clone https://github.com/aaravm10/wc26-predictor.git
cd wc26-predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
