# Breast Cancer Type Prediction Project

This project leverages machine learning algorithms to predict the type of breast cancer (Benign or Malignant) based on various medical features. The application aims to assist in early diagnosis and improve patient outcomes.

## Table of Contents

- [Libraries Used](#libraries-used)
- [Technology](#technology)
- [Model Selection](#model-selection)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Cloning the Repository](#cloning-the-repository)

## Libraries Used

This project utilizes several Python libraries for data manipulation, visualization, and machine learning:

```python
import numpy as np # Numerical Python
import pickle  # To dump & load model in pkl format
import pandas as pd  # Data Manipulation
import seaborn as sns  # Data Visualization
import matplotlib.pyplot as plt  # Data Visualization
from matplotlib.pylab import rcParams as customize_screen  # Setting screen size - PLots size

from sklearn.model_selection import train_test_split  # For splitting data into training & testing
from sklearn.preprocessing import StandardScaler  # For scaling the data - (-x to x)
from sklearn.preprocessing import MinMaxScaler  # For Multinomial NB (0 to 1)

from sklearn.linear_model import LogisticRegression  # Logistic Regression model - Sigmoid function - Good for Binary classification
from sklearn.tree import DecisionTreeClassifier  # Decision Tree model - Iterative Dichotomiser 3 (ID3)
from sklearn.ensemble import RandomForestClassifier  # Random Forest model - Bagging Technique - Bunches of Decision Trees
from sklearn.neighbors import KNeighborsClassifier  # K nearest neighbor model - Distance Formula
from sklearn.svm import SVC  # Support Vector Machine model - Hyperplane
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB  # Naive Bayes model - Bayes Theorem (Conditional Probability)
from xgboost import XGBClassifier  # Extreme Gradient Boosting - Boosting Technique - Penalize the gradients (loss/params)

from sklearn.metrics import accuracy_score  # To calculate the accuracy of the model

customize_screen['figure.figsize'] = 15, 7  # Length, Breadth
```

## Technology

This project is built using **Streamlit**, a powerful library for creating interactive web applications with Python. It provides a simple way to build and deploy machine learning models as web apps.

## Model Selection

Multiple machine learning models were considered for predicting breast cancer types, including:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors Classifier
- Support Vector Classifier (SVC)
- Naive Bayes Classifiers (Gaussian, Multinomial, Bernoulli)
- XGBoost Classifier

After conducting hyperparameter tuning on these models, the following accuracies were achieved:

- **Logistic Regression**: 98.25% (Selected as the primary model)
- **K-Nearest Neighbors**: 96.49%

The models were evaluated based on their accuracy scores, with Logistic Regression emerging as the most reliable option for this dataset.

## Getting Started

1. Clone the repository using the command below.
2. Navigate to the project directory.
3. Install the required dependencies as specified in the `requirements.txt` file.
4. Run the application using your preferred method (e.g., Streamlit).

## Usage

To use the application:

1. Launch the Streamlit app using the command:
   ```bash
   streamlit run app.py
   ```
2. Input the relevant features in the web interface.
3. Click the "Predict" button to see the predicted breast cancer type.

## Cloning the Repository

To clone the repository and specifically access the code in the `master` branch, you can use the following command:

```bash
git clone -b master https://github.com/UmerSalimKhan/BreastCancerTypePrediction_StreamlitEdition.git
```

For any questions or contributions, feel free to open an issue or submit a pull request.
