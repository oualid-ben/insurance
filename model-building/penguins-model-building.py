import pandas as pd
insurance = pd.read_csv('https://raw.githubusercontent.com/oualid-ben/data/main/clean_data_fraud.csv', on_bad_lines='skip')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = insurance.copy()


# Separating X and y
X = df.drop('fraud_reported', axis=1)
Y = df['fraud_reported']

# Build random forest model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 30)
knn.fit(X, Y)

# Saving the model

import pickle
pickle.dump(knn, open('insurance.pkl', 'wb'))
