# Smart Electricity Consumption Prediction

## About the Project

Smart Electricity Consumption Prediction is a simple Machine Learning project that predicts electricity consumption using basic information such as temperature, humidity, previous electricity usage, and hour.

The main purpose of this project is to understand how Machine Learning can be used to predict electricity consumption from given data.

## Objectives

- Predict electricity consumption using Machine Learning.
- Understand how input data affects electricity usage.
- Learn the basic process of creating an ML model.
- Visualize the actual and predicted values.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Dataset

The dataset contains the following columns:

- Temperature
- Humidity
- Previous_Usage
- Hour
- Consumption

The first four columns are used as input features, and Consumption is the value that the model predicts.

## Machine Learning Model

This project uses a Machine Learning regression model because electricity consumption is a continuous numerical value.

The basic steps are:

1. Load the dataset.
2. Select input and output columns.
3. Split the data into training and testing data.
4. Train the Machine Learning model.
5. Make predictions.
6. Compare the predicted values.
7. Display the results using a graph.

## Project Structure

```text
Smart-electricity-consumption-prediction/
│
├── project1.py
├── electricity_prediction.py
├── electricity_data.csv
├── graph.png
└── README.md