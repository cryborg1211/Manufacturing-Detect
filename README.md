# Predictive Maintenance System using PySpark

This project implements a predictive maintenance system designed to detect machine failures using sensor data. It leverages **PySpark** for scalable data processing and machine learning, employing a **Random Forest Classifier** to predict potential failures based on operational metrics.

## 📌 Project Overview

The goal of this project is to predict whether a machine will fail based on various sensor readings. It involves a complete machine learning pipeline including:

1.  **Data Ingestion**: Loading manufacturing sensor data.
2.  **Preprocessing**: Cleaning data, encoding categorical variables, and scaling numerical features.
3.  **Model Training**: Training a Random Forest model with hyperparameter tuning.
4.  **Evaluation**: Assessing model performance using Area Under ROC (AUC) and Confusion Matrix.
5.  **Streaming Simulation**: Simulating a real-time data stream to detect failures as they happen.

## 📂 File Structure

-   `main_code.ipynb`: The main Jupyter Notebook containing the entire pipeline (Data Loading, Preprocessing, Modeling, Evaluation, and Streaming).
-   `machine failure.csv`: The dataset used for training and testing. It contains sensor readings like temperature, speed, torque, and tool wear.
-   `submission.csv`: output sample file from dataset (I don't use in this project)

## 📊 Dataset

The dataset (`machine failure.csv`) includes the following key features:

-   **Type**: Quality type of the product (L, M, H).
-   **Air temperature [K]**: Ambient temperature.
-   **Process temperature [K]**: Temperature during the process.
-   **Rotational speed [rpm]**: Speed of the machine's operation.
-   **Torque [Nm]**: Torque applied.
-   **Tool wear [min]**: Duration of tool usage.
-   **Machine failure**: The target variable (1 = Failure, 0 = No Failure).

## 🛠️ Technology Stack

-   **Python 3**
-   **PySpark** (MLlib for Machine Learning, Structured Streaming for simulation)
-   **Pandas** (For data manipulation in simulation)

## 🚀 How to Run

1.  **Prerequisites**: Ensure you have Python and PySpark installed.
    > **Note**: This notebook is optimized for **Linux** or **Google Colab**. Running on **Windows** may cause environment-related errors with PySpark.
    ```bash
    pip install pyspark pandas
    ```

2.  **Run the Notebook**: Open `main_code.ipynb` in Jupyter Notebook or Google Colab.
    ```bash
    jupyter notebook main_code.ipynb
    ```

3.  **Execute Cells**: Run the cells step-by-step to:
    -   Initialize the Spark Session.
    -   Preprocess the data.
    -   Train and tune the Random Forest model.
    -   Start the real-time streaming simulation to see failure alerts.

## 🔍 Key Features

-   **Pipeline Architecture**: Uses Spark ML Pipelines for modular and reproducible preprocessing and training.
-   **Hyperparameter Tuning**: Optimizes the Random Forest model using Grid Search and Cross-Validation.
-   **Real-time Alerting**: The system simulates a live sensor feed and triggers alerts when a failure is predicted (`prediction = 1.0`).

## 📈 Performance

The model is evaluated using the **Area Under ROC (AUC)** metric to ensure robust classification performance on the unbalanced dataset.

