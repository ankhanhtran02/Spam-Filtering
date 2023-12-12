import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from preprocess import preprocess

X_train_v, X_test_v, y_train, y_test = preprocess(filename='emails.csv')

class LogisticRegressionSolver:
    def __init__(self,X_train_v = X_train_v,X_test_v = X_test_v, y_train = y_train, y_test = y_test):
        model = LogisticRegression(solver='liblinear', C=10.0, random_state=1)
        ''' Penalty: L2 '''
        model.fit(X_train_v, y_train)
        self.y_predict_proba = model.predict_proba(X_test_v)
        self.y_predict = model.predict(X_test_v)
        self.cm = metrics.confusion_matrix(y_test, self.y_predict)
        ''' Report the metrics for each class:'''
        self.report = metrics.classification_report(y_test, self.y_predict)
        ''' Metrics for all instances:'''
        self.accuracy = metrics.accuracy_score(y_test, self.y_predict)
        self.precision = metrics.precision_score(y_test, self.y_predict)
        self.recall = metrics.recall_score(y_test,self.y_predict)
        self.f1 = metrics.f1_score(y_test, self.y_predict)
        ''' Explanation:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) where tp is the number of true positives and fp the number of false positives
        recall = tp / (tp + fn) where fn the number of false negatives
        f1 = 2 * (precision * recall) / (precision + recall)
        support: the number of samples of the true response
        macro average: averaging the unweighted mean per label
        weighted average: averaging the support-weighted mean per label
        '''

if __name__ == '__main__':
    a = LogisticRegressionSolver()
    print(a.cm)