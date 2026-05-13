# System Architecture

## Overview

The WC26 Predictor is built as a sequential pipeline where each stage has a
clear input, a clear output, and is independently testable.

## Stage 1: Data Collection

**Input:** External sources (Kaggle, FIFA, FBref)
**Output:** Raw CSV files in `data/raw/`

### Tier 1 — Current Focus
- Historical match results (results, shootouts, goalscorers)
- FIFA rankings history
- 2026 World Cup fixture list

### Tier 2 — Next Phase
- Current season club stats, squad composition, star players, injuries

### Tier 3 — Final Phase
- Team chemistry, momentum, betting odds, tournament DNA, crowd advantage

Raw data is never modified — always preserved as downloaded.

## Stage 2: Feature Engineering

**Input:** Raw CSV files from `data/raw/`
**Output:** Single cleaned feature store at `data/processed/features.csv`

Steps:
- Standardise team names across all sources
- Handle missing values
- Merge all sources into one unified match record
- Calculate Elo ratings across all historical matches
- Engineer features per match:
  - Win rate last 10 games (overall and vs top 10 ranked opponents)
  - Goals scored and conceded per game (last 10)
  - Clean sheet rate (last 10)
  - World Cup appearances, wins, final appearances historically
  - Head to head record vs opponent
  - Penalty shootout win rate
  - Tournament DNA score (knockout stage performance across all competitions)
- Normalise numerical features

Each match becomes one row with all features for both teams.

## Stage 3: Models

**Input:** `data/processed/features.csv`
**Output:** Win/draw/loss probabilities per match

### Elo Rating System (`src/models/elo.py`)
- Rating maintained per team, updated after every historical match
- Rating difference between two teams maps directly to win probability
- Serves as the core reliable baseline

### XGBoost (`src/models/xgboost_model.py`)
- Gradient boosted tree model
- Incorporates all engineered features
- Trained on pre-2018 matches
- Validated on 2018 and 2022 World Cups

### Neural Network (`src/models/neural_net.py`)
- Feedforward network: Input → 128 → 64 → 32 → 3 outputs
- Leaky ReLU hidden layers, Softmax output
- Adam optimizer with mini-batch gradient descent

### Ensemble (`src/models/ensemble.py`)
- Combines all three model outputs via weighted average
- Starting weights: Elo 40%, XGBoost 40%, Neural Network 20%
- Weights tuned on 2018 World Cup validation set

## Stage 4: Simulation Engine

**Input:** Match probabilities from ensemble
**Output:** 10,000 simulated tournament brackets

Steps:
1. Simulate all 48 team group stage matches
2. Determine group standings and qualified teams
3. Simulate knockout rounds using bracket logic
4. Resolve draws in knockout matches using historical shootout records
5. Repeat 10,000 times and aggregate results

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

## Validation Strategy
Training:   All international matches before 2018
Validation: 2018 World Cup — used for tuning
Test:       2022 World Cup — one final honest check

Baseline to beat: always picking higher FIFA ranked team = ~65% match accuracy