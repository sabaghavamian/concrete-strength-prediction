# Concrete Strength Prediction

A machine learning project for predicting the **compressive strength of concrete** using Python and PyTorch.

## Overview

This project explores the use of a neural network to estimate concrete compressive strength based on concrete mixture characteristics and age.

The workflow includes data preprocessing, feature standardization, train/test splitting, neural network training, model evaluation, and visualization of prediction performance.

## Dataset

The dataset contains **eight input features** related to concrete mixture composition and age, with concrete compressive strength as the target variable.

The dataset file used by the project is:

`Concrete_Data - Concrete_Data.csv`

The model automatically uses the first eight columns as input features and the final column as the prediction target.

## Features

* Data loading and preprocessing with pandas
* Feature standardization using `StandardScaler`
* Reproducible train/test splitting
* Multilayer Perceptron neural network implemented with PyTorch
* Model training using the Adam optimizer
* Performance evaluation using MSE and R²
* Training-loss visualization
* Actual vs. predicted strength visualization
* Prediction-error distribution analysis
* Prediction of compressive strength for new concrete mixtures

## Technologies

* Python
* PyTorch
* pandas
* NumPy
* scikit-learn
* Matplotlib

## Model Architecture

The project uses a **Multilayer Perceptron (MLP)** neural network.

Architecture:

```text
Input Layer: 8 features
        ↓
Hidden Layer: 64 neurons + ReLU
        ↓
Hidden Layer: 32 neurons + ReLU
        ↓
Output Layer: 1 value
```

The output represents the predicted concrete compressive strength.

## Training Configuration

* Train/Test Split: **80% / 20%**
* Random Seed: **42**
* Number of Epochs: **100**
* Optimizer: **Adam**
* Learning Rate: **0.01**
* Loss Function: **Mean Squared Error (MSE)**

A fixed random seed is used to improve the reproducibility of the experiment.

## Evaluation

Model performance is evaluated using:

* **Mean Squared Error (MSE)**
* **R² Score**

The project also uses visual analysis to better understand model performance.

### Training Loss Curve

Shows how the training loss changes during the 100 training epochs.

### Actual vs. Predicted

Compares the predicted concrete compressive strength values with the actual test values.

An ideal prediction would place the points close to the reference diagonal line.

### Error Distribution

Displays the distribution of prediction errors across the test dataset.

## Project Structure

```text
concrete-strength-prediction/
├── Concrete_Data - Concrete_Data.csv
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/sabaghavamian/concrete-strength-prediction.git
cd concrete-strength-prediction
```

### 2. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 3. Check the dataset

Make sure the following file is located in the root directory of the project:

```text
Concrete_Data - Concrete_Data.csv
```

### 4. Run the project

```bash
python main.py
```

The script will:

1. Load the concrete dataset
2. Separate the input features and target
3. Split the data into training and testing sets
4. Standardize the input features
5. Train the PyTorch neural network
6. Calculate MSE and R² on the test set
7. Display the training-loss curve
8. Display the actual-vs.-predicted plot
9. Display the prediction-error distribution
10. Predict compressive strength for sample concrete mixtures

## Requirements

The required Python libraries are listed in `requirements.txt`:

```text
torch
numpy
pandas
scikit-learn
matplotlib
```

## Example Prediction

The trained model can also be used to estimate compressive strength for new concrete-mixture inputs.

Two sample mixtures are included in `main.py` to demonstrate the prediction workflow after feature standardization.

## Purpose

This project was developed as part of my academic work to practice the application of **machine learning and neural networks to civil engineering problems**.

It also represents my broader interest in combining structural and earthquake engineering with data-driven methods and artificial intelligence.

## Future Improvements

Possible future developments include:

* Comparing the neural network with other regression algorithms
* Hyperparameter optimization
* Cross-validation
* Feature-importance analysis
* Improved neural network architectures
* Automated saving of evaluation plots
* Additional statistical analysis of prediction errors
* Application of data-driven methods to Structural Health Monitoring problems

## Author

**Saba Ghavamian**

M.Sc. Candidate in Earthquake Engineering
Iran University of Science and Technology (IUST)

Research interests:

* Structural Health Monitoring (SHM)
* Earthquake Engineering
* Structural Analysis
* Machine Learning
* Data-Driven Civil Engineering

## License

This project is available under the **MIT License**.
