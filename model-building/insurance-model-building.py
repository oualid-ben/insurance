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
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']
dtc = DecisionTreeClassifier()
grid_params = {'criterion': ['gini', 'entropy'],'max_depth': [3, 5, 7, 10],'min_samples_split': range(2, 10, 1),'min_samples_leaf': range(2, 10, 1)}
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.25,random_state= 0)

grid_search = GridSearchCV(dtc, grid_params, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, Y_train)
be = grid_search.best_estimator_

pickle.dump(be, open('insurance_tree_1.pkl', 'wb'))
