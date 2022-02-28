import pandas as pd
insurance = pd.read_csv('https://github.com/oualid-ben/data/blob/main/clean_data_fraud.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = insurance.copy()
target = 'fraud_reported'

target_mapper = {'Pas de fraud':0, 'Fraude':1}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)

# Separating X and y
X = df.drop('fraud_reported', axis=1)
Y = df['fraud_reported']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('insurance_clf.pkl', 'wb'))
