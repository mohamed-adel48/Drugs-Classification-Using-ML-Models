# Drugs-Classification-Using-ML-Models
This repository contains a project for classifying different drugs based on patient information such as age, gender, and medical conditions. The goal is to build a machine learning model that accurately predicts the appropriate drug for patients. The project utilizes a dataset from Kaggle and employs different techniques such as K-Nearest Neighbors (KNN), Decision Trees, Random Forest, Gradient Boosting, and Support Vector Machine (SVM) for the classification tasks. 

## Dataset
The dataset used in this project is sourced from Kaggle: [Drugs A, B, C, X, Y for Decision Trees](https://www.kaggle.com/datasets/pablomgomez21/drugs-a-b-c-x-y-for-decision-trees/data), also available in the `drug200.csv` file. It contains information about various patients and the corresponding drugs they were prescribed. The dataset includes the following features:
- **Age**: The age of the patient.
- **Sex**: The gender of the patient (Male or Female).
- **BP**: Blood pressure levels of the patient (High, Normal, or Low).
- **Cholesterol**: Cholesterol levels of the patient (High or Normal).
- **Na_to_K**: Sodium to potassium ratio in the patient's blood.
- **Drug**: The drug prescribed (DrugY, DrugC, DrugX, DrugA, DrugB).

## Notebook
The Jupyter Notebook `Drugs_Classification.ipynb` contains the code for data preprocessing, model building, and evaluation. The notebook walks through the following steps:

1. **Data Loading**: Importing the dataset and displaying the first few records.
2. **Data Preprocessing**: Handling missing values, encoding categorical features, and splitting the dataset into training and testing sets.
3. **Model Building**: Constructing and training the KNN and Decision Tree models.
4. **Model Evaluation**: Evaluating the performance of the models using metrics such as accuracy, precision, recall, and F1-score.
5. **Results**: Comparing the results of the models and drawing conclusion.

## Model Performance

Below is a table showing the accuracy of different models used in this project:

| Model                  | Accuracy |
|------------------------|----------|
| Random Forest          | 100%     |
| Decision Tree          | 97.5%    |
| Support Vector Machine | 97.5%    |
| Gradient Boosting      | 97.5%    |
| K-Nearest Neighbors    | 65%      |

## Requirements
The project requires the following Python libraries:

- Pandas
- NumPy
- Sci-kit-learn
- Matplotlib
- Seaborn
- Jupyter

You can install the necessary libraries using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
