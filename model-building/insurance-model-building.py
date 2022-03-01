import pandas as pd
from sklearn.model_selection import train_test_split

insurance = pd.read_csv('https://raw.githubusercontent.com/oualid-ben/data/main/clean_data_fraud.csv', on_bad_lines='skip')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = insurance.copy()


# Separating X and y
X = df.drop('fraud_reported', axis=1)
Y = df['fraud_reported']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
        
        
# Build random forest model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)


# Saving the model
import pickle
pickle.dump(knn, open('insurance.pkl', 'wb'))

from sklearn.ensemble import RandomForestClassifier
# Build random forest model
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
pickle.dump(clf, open('insurance_forest.pkl', 'wb'))

