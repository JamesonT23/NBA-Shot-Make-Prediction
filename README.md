# NBA Shot Make Probability Prediction

A machine learning project that predicts the probability of a basketball shot being made using shot-level data, feature engineering, and a CatBoost classification model.

## Project Overview

This project develops an end-to-end machine learning pipeline to estimate shot-make probabilities. The goal is to use shot characteristics and defensive context to identify patterns that influence whether a shot is successful.

The workflow includes data preparation, feature engineering, model training, evaluation, and prediction generation.

## Technologies Used

- **Python**
- **CatBoost**
- **scikit-learn**
- **Pandas**
- **Polars**
- **NumPy**

## Dataset

The project uses separate training and testing datasets containing basketball shot records.

- Training data: 425,719 shot records
- Testing data: 213,977 shot records
- Target variable: Whether a shot was made or missed

## Feature Engineering

Developed and prepared predictive features to capture shot difficulty and defensive context, including:

- Defender distance
- Defensive pressure
- Closing distance
- Shot distance
- Three-point shot indicators

These features help the model identify relationships between shot conditions and the likelihood of a successful attempt.

## Model Development

A CatBoost classification model was trained to estimate the probability of a shot being made.

The model development process included:

1. Preparing and cleaning the training and testing datasets.
2. Engineering features relevant to shot outcomes.
3. Training a CatBoost classification model.
4. Evaluating model performance using log loss.
5. Generating predicted make probabilities for the testing dataset.

The model achieved a validation log loss of **0.620843**, improving on a baseline log loss of **0.65618**.

## Project Workflow

```text
Raw Shot Data
     |
     v
Data Cleaning and Preparation
     |
     v
Feature Engineering
     |
     v
CatBoost Model Training
     |
     v
Model Evaluation
     |
     v
Shot-Make Probability Predictions
     |
     v
CSV Output
```

## Key Skills Demonstrated

- Machine learning and classification
- Predictive modeling
- Feature engineering
- Data cleaning and preprocessing
- Python-based data workflows
- Model evaluation using log loss
- Working with large datasets
- Generating structured prediction outputs

## Project Purpose

This project demonstrates how machine learning can be applied to basketball analytics to estimate shot success probabilities and explore the factors associated with shot outcomes.