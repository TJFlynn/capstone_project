import numpy as np
import pandas as pd
import gc
from sklearn.externals import joblib

print("-------------------------Beginning Prediction-------------------------")

# Load the properties dataset. Low_memory=False needed due to size of the dataset.
prop = pd.read_csv('data/properties_2016.csv', low_memory=False)

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
imputer = joblib.load('transformers/imputer.pkl')
prop[numerical] = imputer.transform(prop[numerical])
print("Missing numerical values filled")

#Load Scaler and apply StandardScaling to numerical values
scaler = joblib.load('transformers/scaler.pkl')
prop[numerical] = scaler.transform(prop[numerical])
print("Numerical Values scaled")

#Load Categorical Imputer and fill missing categorical values
cat_imputer = joblib.load('transformers/cat_imputer.pkl')
prop[categorical] = cat_imputer.transform(prop[categorical])
print("Missing Categorical Values filled")

#Load Zip Code Label Encoder and apply transformation
le_zip = joblib.load('transformers/le_zip.pkl')
prop['regionidzip'] = le_zip.transform(prop['regionidzip'])
print("Zip Code Label Encoded")

#Load Transaction Month Encoder and apply transformation
le_month = joblib.load('transformers/le_month.pkl')
prop['transaction_month'] = le_month.transform(prop['transaction_month'])
print("Transaction Month Label Encoded")


#Clean Up
del numerical, categorical, scaler, imputer, le_zip, cat_imputer
gc.collect()

#load Trained Regressor
model = joblib.load('models/cat.pkl')

#Begin prediction, then load, update, and write submission file     
print("Predicting October 2016")
pred_Oct16 = model.predict(prop)

sub = pd.read_csv('C:/Users/Travis/Downloads/kaggle_competitions/kaggle_zillow/sample_submission.csv')
sub['201610'] = pred_Oct16

del pred_Oct16
gc.collect()

#reset transaction month for November
prop["transaction_month"] = 11
prop['transaction_month'] = le_month.transform(prop['transaction_month'])

print("Predicting November 2016")
pred_Nov16 = model.predict(prop)

sub['201611'] = pred_Nov16
del pred_Nov16
gc.collect()

#reset transaction month for December
prop["transaction_month"] = 12
prop['transaction_month'] = le_month.transform(prop['transaction_month'])

print("Predicting December 2016")
pred_Dec16 = model.predict(prop)

sub['201612'] = pred_Dec16
del pred_Dec16
gc.collect()

#create a regular csv for sanity checks
print('Writing csv ...')
sub.to_csv('sample_submission.csv', index=False, float_format='%.4g')

#create a compressed csv for actual submission
print('Writing compressed csv ...')
sub.to_csv('sample_submission.csv.gz', index=False, float_format='%.4g', compression='gzip')

print("-------------------------Prediction Complete!-------------------------")