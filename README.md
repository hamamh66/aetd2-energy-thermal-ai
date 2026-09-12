# AET-D2: Agentic Energy-Thermal Decision Intelligence

## Overview
AET-D² is a decision-centric framework that integrates renewable energy forecasting with thermodynamic load modeling and intelligent decision-making. It combines ensemble machine learning, entropy-guided feature selection, and thermal-aware optimization to model realistic energy systems under dynamic conditions.

## Key Features
- Ensemble-based energy forecasting (XGBoost, LightGBM, ExtraTrees)
- Entropy-guided feature relevance analysis
- Thermodynamics-inspired thermal load modeling
- Utility-based decision intelligence layer
- Robust cross-validation and ablation study

## Why It Matters
Modern energy systems are increasingly constrained by both renewable intermittency and thermal demand. AET-D² moves beyond prediction by enabling adaptive, explainable, and operationally meaningful decisions under energy stress conditions.

## Outputs
The notebook automatically generates:
- Model benchmarking tables
- Cross-validation results
- Ensemble performance metrics
- Thermal-aware system evaluation
- Ablation study
- Saved figures and tables to Google Drive
- outputs_summary.txt (full experiment log)

## Usage
Run the notebook in Google Colab. Mount Google Drive when prompted. All outputs will be saved automatically.

## Structure
- Data preprocessing
- Feature engineering
- Entropy analysis
- Model benchmarking
- Ensemble learning
- Thermal modeling
- Decision layer
- Ablation study

## License
Open-source for research and educational use.

## Revision R2 additions (Energy Conversion and Management: X, ECMX-D-26-00831)

- `sensitivity_analysis.py` — re-derives the nominal thermal proxy, comfort gap and action assignment from `aetd2_results_first_1000.csv` (asserting exact reproduction), then evaluates the sensitivity of the action distribution to the utility-score coefficients and the thermal-proxy parameters (96 configurations).
- `sensitivity_all.csv`, `sensitivity_utility_weights.csv`, `sensitivity_thermal_proxy.csv`, `sensitivity_summary.csv` — result tables.
- `sensitivity_action_share.png` — Fig. 7 of the revised manuscript.
- `aetd2_action_distribution.png` — regenerated from `action_distribution.csv` (942 / 58).
