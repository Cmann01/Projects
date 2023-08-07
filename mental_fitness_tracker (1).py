# Import necessary libraries for Azure
import azureml.core
from azureml.core import Workspace, Experiment
from azureml.opendatasets import NycTlcGreen

# Check Azure ML SDK version
print("Azure ML SDK Version:", azureml.core.VERSION)

# Authenticate with Azure (You will need to provide your Azure subscription ID and resource group)
subscription_id = '978b8bd5-3e80-4353-b170-2be776b23d78'
resource_group = 'Projects'
workspace_name = 'Mental_fitness'

# Load the Azure ML workspace
workspace = Workspace(subscription_id, resource_group, workspace_name)

# Load data using Azure Open Datasets
nyc_taxi_dataset = NycTlcGreen(workspace=workspace)

# Convert data to a Pandas DataFrame
df = nyc_taxi_dataset.to_pandas_dataframe()

# Continue with your data processing, cleaning, and model training code
# ... (Replace df with data in your code as mentioned before)

# <Your Data Preprocessing and Model Training Code Here>

# Split data into features (x) and target (y)
x = df.drop('mental fitness', axis=1)
y = df['mental fitness']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# <Your Model Training and Evaluation Code Here>

# Train your model (e.g., using Linear Regression)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(x_train, y_train)

# Model evaluation for training set
y_train_pred = lr.predict(x_train)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

print("The Linear Regression model performance for training set")
print("-------------------------------")
print('MSE is {}'.format(mse_train))
print('RMSE is {}'.format(rmse_train))
print('R2 score is {}'.format(r2_train))

# Model evaluation for testing set
y_test_pred = lr.predict(x_test)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

print("The Linear Regression model performance for testing set")
print("--------------------------------------------")
print('MSE is {}'.format(mse_test))
print('RMSE is {}'.format(rmse_test))
print('R2 score is {}'.format(r2_test))

# <Your Random Forest Regression Model Training and Evaluation Code Here>

# Submit the experiment run to Azure Machine Learning
experiment_name = 'mental-fitness-prediction-experiment'
exp = Experiment(workspace, experiment_name)
run = exp.start_logging()

# Log metrics for Linear Regression
run.log('Linear Regression - MSE', mse_test)
run.log('Linear Regression - RMSE', rmse_test)
run.log('Linear Regression - R2 Score', r2_test)

# ... (Log metrics for Random Forest Regression if used)

run.complete()
