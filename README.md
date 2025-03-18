# Diabetes Prediction Application

## Overview
This project is a machine learning application that predicts the likelihood of diabetes based on various health metrics. It consists of a data analysis script and a user-friendly web application built with Streamlit.

## Features
- Machine learning model using Random Forest algorithm
- Interactive web interface for user input
- Visualizations of model performance and data correlations
- Real-time prediction based on user-provided health metrics

## Dataset
The application uses the Pima Indians Diabetes Dataset which includes the following features:
- Pregnancies
- Glucose level
- Blood pressure
- Skin thickness
- Insulin level
- BMI (Body Mass Index)
- Diabetes pedigree function
- Age
- Outcome (target variable: 1 for diabetes, 0 for no diabetes)

## Project Structure
- `assignment.py`: Contains the data analysis, model training and evaluation code
- `app.py`: Streamlit web application for user interaction
- `diabetes.csv`: The dataset used for model training

## Setup and Running the Application
1. Install the required dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn streamlit
   ```

2. Run the data analysis and model training:
   ```
   python assignment.py
   ```

3. Launch the web application:
   ```
   streamlit run app.py
   ```

## Model Performance
The Random Forest classifier is trained on 80% of the data and evaluated on the remaining 20%. Performance metrics like accuracy, precision, recall, and F1-score are calculated and displayed when running the `assignment.py` script.