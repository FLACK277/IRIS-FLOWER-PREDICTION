# IRIS-SPECIES-PREDICTION
A machine learning project that builds classification models to identify iris flower species based on their sepal and petal measurements.
Overview
This project implements and evaluates multiple classification algorithms to predict iris flower species (setosa, versicolor, and virginica) using the classic Iris dataset. The implementation includes data exploration, preprocessing, model training, hyperparameter tuning, and feature importance analysis.
Features

Data Exploration & Visualization: Comprehensive data analysis with pairplots, boxplots, and correlation heatmaps
Model Comparison: Implementation and evaluation of multiple classification algorithms:

Logistic Regression
Decision Tree
Random Forest
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)


Model Evaluation: Performance metrics including accuracy, precision, recall, and F1-score
Hyperparameter Tuning: Grid search optimization for the best performing model
Feature Importance Analysis: Identification of the most significant features for classification
Prediction Function: Practical implementation for classifying new iris samples

Requirements

Python 3.6+
Required libraries:
pandas
numpy
matplotlib
seaborn
scikit-learn


Installation

Clone this repository:
bashgit clone https://github.com/yourusername/iris-classification.git
cd iris-classification

Install the required dependencies:
bashpip install -r requirements.txt


Dataset
The Iris dataset contains 150 samples of iris flowers, with the following features:

sepal_length: Length of the sepal in cm
sepal_width: Width of the sepal in cm
petal_length: Length of the petal in cm
petal_width: Width of the petal in cm
species: The species of iris (setosa, versicolor, or virginica)

Usage
Run the main script to execute the full classification pipeline:
bashpython iris_classification.py
When prompted, you can either:

Provide a path to your own iris dataset CSV file
Press Enter to use the built-in iris dataset

Files

iris_classification.py: Main script with all implemented functionality
iris_classification_model.pkl: Saved best model (created after running the script)
iris_pairplot.png: Visualization of pairwise relationships between features
iris_boxplots.png: Boxplots showing feature distributions by species
iris_correlation.png: Correlation heatmap of features
confusion_matrix_*.png: Confusion matrices for each model
feature_importance.png: Visualization of feature importance

Key Findings

Most Significant Features: Petal length and petal width are typically the most important features for distinguishing between iris species
Model Performance: SVM and Random Forest models often achieve the highest accuracy (95-98%)
Species Separability: Setosa is easily distinguishable, while versicolor and virginica show some overlap


Data Loading and Exploration:

Load the iris dataset
Examine basic statistics and class distribution
Check for missing values


Data Visualization:

Create pairplots to visualize relationships between features
Generate boxplots to understand feature distributions by species
Build correlation heatmaps to identify feature relationships


Data Preprocessing:

Split data into training and testing sets (80/20 split)
Apply feature scaling using StandardScaler


Model Building and Evaluation:

Train multiple classification models
Evaluate each model using accuracy, classification report, and confusion matrix
Select the best performing model


Hyperparameter Tuning:

Perform grid search to optimize the best model
Re-evaluate the tuned model


Feature Importance Analysis:

Determine which features contribute most to classification
Visualize feature importance


Model Saving and Prediction:

Save the final model for future use
Provide a function for making predictions on new data



Contributing

Fork the repository
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

The Iris dataset was introduced by the British statistician and biologist Ronald Fisher in 1936
This project uses scikit-learn for machine learning algorithms
