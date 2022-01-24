!pip3 install numpy
!pip3 install pandas
!pip3 install scikit-learn
!pip3 install mlflow
!git+https://github.com/fastforwardlabs/cmlbootstrap#egg=cmlbootstrap
!pip3 install dill

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

#from your_data_loader import load_data
from churnexplainer import CategoricalEncoder


import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import xml.etree.ElementTree as ET
from cmlbootstrap import CMLBootstrap
# Set the setup variables needed by CMLBootstrap
HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT")  

# Instantiate API Wrapper
cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)


spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .master("local[*]")\
    .getOrCreate()

# **Note:**
# Our file isn't big, so running it in Spark local mode is fine but you can add the following config
# if you want to run Spark on the kubernetes cluster
#
# > .config("spark.yarn.access.hadoopFileSystems",os.getenv['STORAGE'])\
#
# and remove `.master("local[*]")\`
#

# Since we know the data already, we can add schema upfront. This is good practice as Spark will
# read *all* the Data if you try infer the schema.

schema = StructType(
    [
        StructField("customerID", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("SeniorCitizen", StringType(), True),
        StructField("Partner", StringType(), True),
        StructField("Dependents", StringType(), True),
        StructField("tenure", DoubleType(), True),
        StructField("PhoneService", StringType(), True),
        StructField("MultipleLines", StringType(), True),
        StructField("InternetService", StringType(), True),
        StructField("OnlineSecurity", StringType(), True),
        StructField("OnlineBackup", StringType(), True),
        StructField("DeviceProtection", StringType(), True),
        StructField("TechSupport", StringType(), True),
        StructField("StreamingTV", StringType(), True),
        StructField("StreamingMovies", StringType(), True),
        StructField("Contract", StringType(), True),
        StructField("PaperlessBilling", StringType(), True),
        StructField("PaymentMethod", StringType(), True),
        StructField("MonthlyCharges", DoubleType(), True),
        StructField("TotalCharges", DoubleType(), True),
        StructField("Churn", StringType(), True)
    ]
)

# Now we can read in the data from Cloud Storage into Spark...
try : 
  storage=os.environ["STORAGE"]
except:
  if os.path.exists("/etc/hadoop/conf/hive-site.xml"):
    tree = ET.parse('/etc/hadoop/conf/hive-site.xml')
    root = tree.getroot()
    for prop in root.findall('property'):
      if prop.find('name').text == "hive.metastore.warehouse.dir":
        storage = prop.find('value').text.split("/")[0] + "//" + prop.find('value').text.split("/")[2]
  else:
    storage = "/user/" + os.getenv("HADOOP_USER_NAME")
  storage_environment_params = {"STORAGE":storage}
  storage_environment = cml.create_environment_variable(storage_environment_params)
  os.environ["STORAGE"] = storage


storage = os.environ['STORAGE']
hadoop_user = os.environ['HADOOP_USER_NAME']

telco_data = spark.read.csv(
    "{}/user/{}/data/churn/WA_Fn-UseC_-Telco-Customer-Churn-.csv".format(
        storage,hadoop_user),
    header=True,
    schema=schema,
    sep=',',
    nullValue='NA'
)

# ...and inspect the data.

telco_data.show()

telco_data.printSchema()

# Now we can store the Spark DataFrame as a file in the local CML file system
# *and* as a table in Hive used by the other parts of the project.

telco_data.coalesce(1).write.csv(
    "file:/home/cdsw/raw/telco-data/",
    mode='overwrite',
    header=True
)

spark.sql("show databases").show()

spark.sql("show tables in default").show()

# Create the Hive table
# This is here to create the table in Hive used be the other parts of the project, if it
# does not already exist.

if ('telco_churn' not in list(spark.sql("show tables in default").toPandas()['tableName'])):
    print("creating the telco_churn database")
    telco_data\
        .write.format("parquet")\
        .mode("overwrite")\
        .saveAsTable(
            'default.telco_churn'
        )

# Show the data in the hive table
spark.sql("select * from default.telco_churn").show()

# To get more detailed information about the hive table you can run this:
df = spark.sql("SELECT * FROM default.telco_churn").toPandas()


idcol = 'customerID'
labelcol = 'Churn'
cols = (('gender', True),
        ('SeniorCitizen', True),
        ('Partner', True),
        ('Dependents', True),
        ('tenure', False),
        ('PhoneService', True),
        ('MultipleLines', True),
        ('InternetService', True),
        ('OnlineSecurity', True),
        ('OnlineBackup', True),
        ('DeviceProtection', True),
        ('TechSupport', True),
        ('StreamingTV', True),
        ('StreamingMovies', True),
        ('Contract', True),
        ('PaperlessBilling', True),
        ('PaymentMethod', True),
        ('MonthlyCharges', False),
        ('TotalCharges', False))


df = df.replace(r'^\s$', np.nan, regex=True).dropna().reset_index()
df.index.name = 'id'
data, labels = df.drop(labelcol, axis=1), df[labelcol]
data = data.replace({'SeniorCitizen': {1: 'Yes', 0: 'No'}})
# This is Mike's lovely short hand syntax for looping through data and doing useful things. I think if we started to pay him by the ASCII char, we'd get more readable code.
data = data[[c for c, _ in cols]]
catcols = (c for c, iscat in cols if iscat)
for col in catcols:
    data[col] = pd.Categorical(data[col])
labels = (labels == 'Yes')

ce = CategoricalEncoder()
X = ce.fit_transform(data)


y=labels.values


#mlflow.set_tracking_uri('http://your.mlflow.url:5000')
mlflow.set_experiment('Sample Experiment')

#X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="run1") as run: 
    # tracking run parameters
    mlflow.log_param("compute", 'local')
    mlflow.log_param("dataset", 'telco-churn')
    mlflow.log_param("dataset_version", '2.0')
    mlflow.log_param("algo", 'random forest')
    
    # tracking any additional hyperparameters for reproducibility
    n_estimators = 5
    mlflow.log_param("n_estimators", n_estimators)

    # train the model
    rf = RandomForestRegressor(n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # automatically save the model artifact to the S3 bucket for later deployment
    mlflow.sklearn.log_model(rf, "rf-baseline-model")

    # log model performance using any metric
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("mse", mse)
    
    mlflow.end_run()
    
    
with mlflow.start_run(run_name="run2") as run: 
    # tracking run parameters
    mlflow.log_param("compute", 'local')
    mlflow.log_param("dataset", 'telco-churn')
    mlflow.log_param("dataset_version", '2.0')
    mlflow.log_param("algo", 'random forest')
    
    # tracking any additional hyperparameters for reproducibility
    n_estimators = 3
    mlflow.log_param("n_estimators", n_estimators)

    # train the model
    rf = RandomForestRegressor(n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # automatically save the model artifact to the S3 bucket for later deployment
    mlflow.sklearn.log_model(rf, "rf-baseline-model")

    # log model performance using any metric
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("mse", mse)
    
    mlflow.end_run()