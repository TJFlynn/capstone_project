import numpy as np
import pandas as pd
import gc

print("-------------------------Beginning Training-------------------------")

# Load the properties 2016 dataset. Low_memory=False needed due to size of the dataset.
properties = pd.read_csv('data/properties_2016.csv', low_memory=False)

# Load the training 2016 dataset which contains the label to train on, logerror.
training = pd.read_csv('data/train_2016_v2.csv', parse_dates=['transactiondate'])

# Use pandas merge function to join the training data and the properties on the unique identifier 'parcelid'
data = training.merge(properties, how='left', on='parcelid')

del training
gc.collect()

# Create a new column for the transaction month from the original transactiondate column
data['transaction_month'] = data['transactiondate'].dt.month

# Select the most useful features
data = data[['logerror', 'calculatedbathnbr', 'bedroomcnt', 'calculatedfinishedsquarefeet',
             'latitude', 'longitude', 'lotsizesquarefeet', 'yearbuilt',
             'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
             'landtaxvaluedollarcnt', 'taxamount', 'regionidzip',
             'transaction_month']]

# Compress the log error of outliers
for idx in data.index:
    q = data.get_value(idx, 'logerror')
    if q > 0.4:
        x = q / 2
        data.set_value(idx, 'logerror', x)
    elif q < -0.4:
        x = q / 2
        data.set_value(idx, 'logerror', x)

# Create target and drop it from the training data
label = data["logerror"]
data = data.drop(['logerror'], 1)

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.externals import joblib

# Define, impute, and transform numerical features
numerical = ['calculatedbathnbr', "bedroomcnt", "calculatedfinishedsquarefeet", "latitude",
            "longitude", "lotsizesquarefeet", "yearbuilt", "structuretaxvaluedollarcnt",
            "taxvaluedollarcnt", "landtaxvaluedollarcnt", "taxamount"]

imputer = Imputer(missing_values='NaN', strategy='median', axis=0)

imputer = imputer.fit(data[numerical])
data[numerical] = imputer.transform(data[numerical])
joblib.dump(imputer, 'transformers/imputer.pkl')
print("Numerical Fillna Complete")

scaler = StandardScaler()
      
scaler = scaler.fit(data[numerical])    
data[numerical] = scaler.transform(data[numerical])
joblib.dump(scaler, 'transformers/scaler.pkl')
print("Numerical Scaling Complete")

# Define, impute, and transform categorical features
categorical = ['regionidzip', "transaction_month"]

cat_imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

cat_imputer = cat_imputer.fit(data[categorical])
data[categorical] = cat_imputer.transform(data[categorical])
joblib.dump(cat_imputer, 'transformers/cat_imputer.pkl')
print("Categorical fillna complete")

from sklearn.preprocessing import LabelEncoder

le_zip = LabelEncoder()
zip_values = properties['regionidzip'].append(data['regionidzip'])
le_zip = le_zip.fit(zip_values.values)
data['regionidzip'] = le_zip.transform(data['regionidzip'])
joblib.dump(le_zip, 'transformers/le_zip.pkl')

print("Zip Code Label Encoded")

le_month = LabelEncoder()
le_month = le_month.fit(data['transaction_month'].values)
data['transaction_month'] = le_month.transform(data['transaction_month'])
joblib.dump(le_month, 'transformers/le_month.pkl')

print("Transaction Month Label Encoded")

del properties
gc.collect()

from catboost import CatBoostRegressor

print("Beginning 2016 Training")
cat = CatBoostRegressor(iterations=1000,
                        learning_rate=0.005,
                        depth=4, l2_leaf_reg=15,
                        loss_function='MAE',
                        eval_metric='MAE',
                        random_seed=10)

cat.fit(data, label)
joblib.dump(cat, 'models/cat.pkl')

print("-------------------------Training Complete-------------------------")