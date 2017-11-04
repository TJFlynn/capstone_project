import numpy as np
import pandas as pd
import gc
from sklearn.externals import joblib

print("-------------------------Beginning Prediction-------------------------")

# Load the properties 2017 dataset. Low_memory=False needed due to size of the dataset.
prop = pd.read_csv('data/properties_2017.csv', low_memory=False)

prop["transaction_month"] = 10


prop = prop[['calculatedbathnbr', 'bedroomcnt', 'calculatedfinishedsquarefeet',
                         'latitude', 'longitude', 'lotsizesquarefeet', 'yearbuilt',
                         'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
                         'landtaxvaluedollarcnt', 'taxamount', 'regionidzip',
                         'transaction_month']]

numerical = ['calculatedbathnbr', "bedroomcnt", "calculatedfinishedsquarefeet", "latitude",
            "longitude", "lotsizesquarefeet", "yearbuilt", "structuretaxvaluedollarcnt",
            "taxvaluedollarcnt", "landtaxvaluedollarcnt", "taxamount"]

categorical = ['regionidzip', "transaction_month"]


#Load Imputer and fill missing numerical values
imputer = joblib.load('transformers/imputer17.pkl')
prop[numerical] = imputer.transform(prop[numerical])
print("Missing numerical values filled")

#Load Scaler and apply StandardScaling to numerical values
scaler = joblib.load('transformers/scaler17.pkl')
prop[numerical] = scaler.transform(prop[numerical])
print("Numerical Values scaled")

#Load Categorical Imputer and fill missing categorical values
cat_imputer = joblib.load('transformers/cat_imputer17.pkl')
prop[categorical] = cat_imputer.transform(prop[categorical])
print("Missing Categorical Values filled")

#Load Zip Code Label Encoder and apply transformation
le_zip = joblib.load('transformers/le_zip17.pkl')
prop['regionidzip'] = le_zip.transform(prop['regionidzip'])
print("Zip Code Label Encoded")

#Load Transaction Month Encoder and apply transformation - 2016 month encoder needed since
#2017 training data doesn't include a single entry for Oct, Nov, or Dec.
le_month = joblib.load('transformers/le_month.pkl')
prop['transaction_month'] = le_month.transform(prop['transaction_month'])
print("Transaction Month Label Encoded")


#Clean Up
del numerical, categorical, scaler, imputer, le_zip, cat_imputer
gc.collect()

#load Trained Regressor
model = joblib.load('models/cat17.pkl')

#Begin prediction, then load, update, and write submission file     
print("Predicting October 2017")
pred_Oct17 = model.predict(prop)

sub = pd.read_csv('C:/Users/Travis/Downloads/kaggle_competitions/kaggle_zillow/sample_submission.csv')
sub['201710'] = pred_Oct17

del pred_Oct17
gc.collect()
for idx in sub.index:
    q = sub.get_value(idx, '201610')
    y = sub.get_value(idx, '201710')
    z = (q + y) / 2
    sub.set_value(idx, '201710', z)

#reset transaction month for November
prop["transaction_month"] = 11
prop['transaction_month'] = le_month.transform(prop['transaction_month'])

print("Predicting November 2017")
pred_Nov17 = model.predict(prop)

sub['201711'] = pred_Nov17
del pred_Nov17
gc.collect()

for idx in sub.index:
    q = sub.get_value(idx, '201611')
    y = sub.get_value(idx, '201711')
    z = (q + y) / 2
    sub.set_value(idx, '201711', z)


#reset transaction month for December
prop["transaction_month"] = 12
prop['transaction_month'] = le_month.transform(prop['transaction_month'])

print("Predicting December 2017")
pred_Dec17 = model.predict(prop)

sub['201712'] = pred_Dec17
del pred_Dec17
gc.collect()

for idx in sub.index:
    q = sub.get_value(idx, '201612')
    y = sub.get_value(idx, '201712')
    z = (q + y) / 2
    sub.set_value(idx, '201712', z)

#create a regular csv for sanity checks
print('Writing csv ...')
sub.to_csv('sample_submission.csv', index=False, float_format='%.4g')

#create a compressed csv for actual submission
print('Writing compressed csv ...')
sub.to_csv('sample_submission.csv.gz', index=False, float_format='%.4g', compression='gzip')

print("-------------------------Prediction Complete!-------------------------")