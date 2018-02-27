import random, math
import pandas as pd
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import manifold
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from pandas_ml import ConfusionMatrix
from sklearn.externals import joblib

def ProcessDataset(pathname):
	stockdata = pd.read_csv(pathname)
	stockdata.dropna(axis=0,how='any',inplace=True)
	OneyearStatus = stockdata['ReturnLabel'].copy()
	OneyearStatus = OneyearStatus.map({'Profit':1,'Not Profit': 0})
	columns_to_drop = ['Stock','ReturnLabel','PE Ratio Group','PB Ratio Group','Year','YearEndReturn']
	stockdata.drop(columns_to_drop,inplace = True, axis = 1)
	stockdata = pd.get_dummies(stockdata,['Sector','SubSector'])
	return stockdata, OneyearStatus

def GetSplits(stockdata, OneyearStatus):
    T = preprocessing.KernelCenterer().fit_transform(stockdata)
    model = Isomap(n_neighbors=5, n_components=2)
    X = model.fit_transform(T)
    x_train, x_test,y_train, y_test = train_test_split(X,OneyearStatus,test_size = 0.2, random_state = 7)
    return x_train, x_test,y_train, y_test

def train_test_and_evaluate(pipeline, x_train, y_train, x_test, y_test):
    pipeline.fit(x_train, y_train)
    y_pred_class = pipeline.predict(x_test)
    confusion_matrix = ConfusionMatrix(list(y_test), list(y_pred_class))
    #display_accuracy_difference(y_test, y_pred_class)
    classification_report = confusion_matrix.classification_report
      
    return pipeline, confusion_matrix

def MakeClassifier(x_train, y_train, x_test, y_test):
	classifier = KNeighborsClassifier(3)

	classifier_pipeline = Pipeline([
        ('classifier', classifier)
    ])
	classifier_pipeline, confusion_matrix = train_test_and_evaluate(classifier_pipeline, x_train, y_train, x_test, y_test)
	joblib.dump(classifier_pipeline, 'C:/API/Model/model.pkl')


if __name__ == '__main__':
	stockdata, OneyearStatus = ProcessDataset("C:/API/StockFundametalData.csv")
	x_train, x_test,y_train, y_test = GetSplits(stockdata,OneyearStatus)
	MakeClassifier (x_train, y_train, x_test, y_test)
	print("Successfully Pickled Model")