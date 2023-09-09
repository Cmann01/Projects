# Mental Fitness Prediction Project

This repository contains code for a machine learning project that predicts mental fitness using Azure Machine Learning services.

## Project Overview

In this project, we use Microsoft Azure services to build a predictive model for mental fitness. We utilize the following Azure services:

- **Azure Machine Learning:** We leverage Azure Machine Learning to create a workspace, load data from Azure Open Datasets, train machine learning models, and log experiment metrics.

- **Azure Open Datasets:** We fetch New York City Green Taxi data using Azure Open Datasets, which is later used for training and testing the machine learning model.

## Code Overview

The code in this repository performs the following tasks:

- Import necessary libraries and authenticate with Azure.
- Load data using Azure Open Datasets and convert it into a Pandas DataFrame.
- Perform data preprocessing, cleaning, and model training.
- Split data into training and testing sets.
- Train a Linear Regression model and evaluate its performance.
- Submit the experiment run to Azure Machine Learning and log metrics.

## Usage

To run this code, you'll need:

- An Azure subscription ID and resource group.
- An Azure Machine Learning workspace named "Mental_fitness."

Update the `subscription_id`, `resource_group`, and `workspace_name` variables with your Azure information.

## Dependencies

The code relies on the following Python libraries and Azure services:

- `azureml-core`
- `azureml-opendatasets`
- `scikit-learn`
- `numpy`

## Running the Code

1. Clone this repository to your local machine.
2. Set up the necessary Azure services and credentials.
3. Run the code by executing the Python script.

## Results

- The code trains a Linear Regression model and logs performance metrics such as MSE, RMSE, and R2 score.
- These metrics can be used to evaluate the model's performance.

## Contributing

Contributions to this project are welcome. Feel free to submit issues and pull requests.

---

Developed by Chinmay
