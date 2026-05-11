# System Architecture

## Overview

The WC26 Predictor is built as a sequential pipeline where each stage has a
clear input, a clear output, and is independently testable.

## Stage 1: Data Collection

**Input:** External sources (Kaggle, FIFA, FBref, betting sites)
**Output:** Raw CSV files in `data/raw/`

Sources:
- Historical match results — Kaggle international football results dataset
- FIFA rankings history — FIFA website
- Player stats — FBref.com
- Betting odds — historical and current odds

Each data source has its own collection script in `src/data/collect.py`
Raw data is never modified — always preserved as downloaded.

## Stage 2: Data Processing

**Input:** Raw CSV files from `data/raw/`
**Output:** Single cleaned feature store at `data/processed/features.csv`

Steps:
- Standardise team names across all sources
- Handle missing values
- Merge all sources into one unified match record
- Engineer features (form, head to head, ranking difference)
- Calculate Elo ratings across all historical matches
- Normalise numerical features

Each match becomes one row with all features for both teams.

## Stage 3: Models

**Input:** `data/processed/features.csv`
**Output:** Win/draw/loss probabilities per match

Three parallel models:

### Elo Rating System (`src/models/elo.py`)
- Maintains a rating per team updated after every historical match
- Rating difference between teams directly maps to win probability
- Serves as the core baseline

### XGBoost (`src/models/xgboost_model.py`)
- Gradient boosted tree model
- Incorporates all engineered features including gut feeling data
- Trained on historical matches pre-2018
- Validated on 2018 and 2022 World Cups

### Neural Network (`src/models/neural_net.py`)
- Feedforward network: Input → 128 → 64 → 32 → 3 outputs
- Same architecture as MNIST baseline, adapted for match prediction
- Leaky ReLU hidden layers, Softmax output
- Adam optimizer with mini-batch gradient descent

### Ensemble (`src/models/ensemble.py`)
- Combines all three model outputs via weighted average
- Weights determined by validation performance on 2018 World Cup
- Default starting weights: Elo 40%, XGBoost 40%, Neural Net 20%

## Stage 4: Simulation Engine

**Input:** Match probabilities from ensemble
**Output:** 10,000 simulated tournament brackets

Steps:
1. Simulate all 48 group stage matches
2. Determine group standings and qualified teams
3. Simulate knockout rounds using bracket logic
4. Handle draws — renormalise to win/loss for knockout matches
5. Repeat 10,000 times
6. Aggregate results

## Stage 5: Output

**Input:** 10,000 simulated brackets
**Output:** Probability distributions and most likely bracket

Per team:
- % chance of winning tournament
- % chance of reaching final
- % chance of reaching semi finals
- % chance of exiting group stage

Per match:
- Win/draw/loss probabilities
- Most likely scoreline

## Validation Strategy
Training:   All international matches before 2018
Validation: 2018 World Cup (used for tuning)
Test:       2022 World Cup (one final check only)

Baseline to beat: always picking higher FIFA ranked team = ~65% match accuracy

## Tech Stack

- Python 3.12
- numpy, pandas — data processing
- xgboost — gradient boosting model
- scikit-learn — preprocessing and evaluation metrics
- matplotlib, seaborn — visualization
- pytest — unit testing
