import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

insurance = pd.read_csv('https://raw.githubusercontent.com/oualid-ben/data/main/clean_data_fraud.csv', on_bad_lines='skip')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = insurance.copy()


# Separating X and y
X = df.drop('fraud_reported', axis=1)
Y = df['fraud_reported']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
        
        
# Build KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)


# Saving the model
import pickle
pickle.dump(knn, open('insurance.pkl', 'wb'))

# Build log model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Saving the model
pickle.dump(logreg, open('insurance_log.pkl', 'wb'))


#building decision tree
# Separating X and y
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

pickle.dump(dtc, open('insurance_tree_1.pkl', 'wb'))
