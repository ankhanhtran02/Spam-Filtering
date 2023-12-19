import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from random import randint
from algorithms import *
from sklearn.decomposition import TruncatedSVD

# Preprocessing data from file
vect = TfidfVectorizer(stop_words='english')
def preprocess(vectorizer, filename='emails.csv'):
    dataset1_df = pd.read_csv(filename)
    X = dataset1_df['text'].astype(str)
    y = dataset1_df['spam']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_train_v = vectorizer.fit_transform(X_train)
    X_test_v = vectorizer.transform(X_test)
    # X_array = X_train_v.toarray()
    # arr_df = pd.DataFrame(X_array, columns=vect.get_feature_names_out())
    return X_train_v, X_test_v, y_train, y_test
X_train_v, X_test_v, y_train, y_test = preprocess(vect, 'emails.csv')

'''
KNN Model with SVD and using MaxAbsScaler
'''

scaler = MaxAbsScaler()
X_train_v_scaled = scaler.fit_transform(X_train_v)
X_test_v_scaled = scaler.transform(X_test_v)

# Perform SVD
svd = TruncatedSVD(100)
X_trainSVD = svd.fit_transform(X_train_v_scaled)
X_testSVD = svd.transform(X_test_v_scaled)

knn = KNeighborsClassifier()

# Tuning hyperparameters by using cross-validation method
parameters = {'n_neighbors': [3, 5, 7, 9, 11, 13], 'metric': ['euclidean', 'manhattan']}
gridSearch = GridSearchCV(knn, param_grid=parameters, cv=5)
gridSearch.fit(X_trainSVD, y_train)
print("Best parameters:", gridSearch.best_params_)
print("Best training Score:", gridSearch.best_score_)

# Evaluate the model on testing set
test_pred = gridSearch.predict(X_testSVD)
print('Testing Set Accuracy:', accuracy_score(y_test, test_pred))
# Printing classification report
print(classification_report(y_test, test_pred))