# Concrete Strength Prediction

A machine learning project for predicting the compressive strength of concrete using Python and PyTorch.

## Overview

This project explores the use of machine learning and neural networks to estimate concrete compressive strength based on mixture characteristics.

The goal is to build a reproducible data-driven workflow including data preprocessing, model training, evaluation, and visualization.

## Features

* Data preprocessing and feature standardization
* Train/test data splitting
* Neural network implementation using PyTorch
* Model training with Adam optimizer
* Performance evaluation using MSE and R²
* Visualization of training loss
* Predicted vs. actual strength comparison
* Error analysis

## Technologies

* Python
* PyTorch
* pandas
* NumPy
* scikit-learn
* Matplotlib

## Model

A multilayer perceptron (MLP) neural network is used to predict concrete compressive strength based on concrete mixture input variables.

## Evaluation

Model performance is evaluated using:

* Mean Squared Error (MSE)
* R² Score

Additional visualizations are used to analyze prediction accuracy and error distribution.

## Project Structure

```text
concrete-strength-prediction/
├── data/
├── src/
├── figures/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

## Author

**Saba Ghavamian**
M.Sc. Candidate in Earthquake Engineering
Iran University of Science and Technology

Research interests: Structural Health Monitoring, Earthquake Engineering, Machine Learning, and Data-Driven Civil Engineering.
