#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import preprocess

X_train_v, X_test_v, y_train, y_test = preprocess(filename='emails.csv')

model = MultinomialNB()
model.fit(X_train_v, y_train)
y_predict_proba = model.predict_proba(X_test_v)
y_predict = model.predict(X_test_v)
score = model.score(X_test_v, y_test)
cm = confusion_matrix(y_test, y_predict)
report = classification_report(y_test, y_predict)

print(report)

