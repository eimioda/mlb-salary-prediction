# MLB Salary Prediction Using Machine Learning

A machine learning project exploring whether historical MLB batting performance can be used to predict player salaries.

## Project Overview

Major League Baseball teams make complex decisions when evaluating player contracts. This project investigates whether player performance statistics and age can be used to predict salary using supervised machine learning.

The analysis uses historical MLB data from **1985–2015**, combining batting statistics, salary information, and player information. The final dataset contains **27,385 player-season observations**.

## Methods

The project follows a typical machine learning workflow:

* Merged batting, salary, and player datasets
* Created player age as a derived feature
* Imputed missing batting statistics with 0
* Applied a `log1p` transformation to salary to address its highly right-skewed distribution
* Selected eight player features for modeling:

  * Home runs
  * RBI
  * Walks
  * Strikeouts
  * Stolen bases
  * Games played
  * At-bats
  * Age
* Split the data into training and test sets
* Used 5-fold cross-validation and `GridSearchCV` for hyperparameter tuning
* Compared three regression models:

  * Lasso Regression
  * Random Forest
  * k-Nearest Neighbors
* Evaluated models using RMSE
* Examined Random Forest feature importance

## Model Performance

| Model               | Cross-Validated RMSE | Test RMSE |
| ------------------- | -------------------: | --------: |
| Lasso Regression    |               1.1684 |    1.1523 |
| Random Forest       |               1.0496 |    1.0392 |
| k-Nearest Neighbors |               1.1103 |    1.1058 |

Random Forest produced the lowest cross-validated and test RMSE among the three models.

## Feature Importance

The Random Forest model's most important features were:

1. Age
2. RBI
3. At-bats
4. Games played
5. Strikeouts

Age had substantially higher feature importance than the other variables in the final Random Forest model.

## Key Takeaways

* MLB salary is influenced by multiple factors rather than a single batting statistic.
* The highly skewed salary distribution required transformation before modeling.
* Random Forest captured nonlinear relationships that the simpler Lasso model could not capture as effectively.
* Age was the most important feature in the Random Forest model, followed by RBI, at-bats, games played, and strikeouts.
* The model demonstrates how machine learning can be used to explore salary patterns, but historical performance statistics alone do not capture all of the factors involved in MLB compensation.

## Tools

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter / Google Colab


This project was originally completed as a final project for **DACSS 756: Machine Learning for Social Scientists** at the University of Massachusetts Amherst.
