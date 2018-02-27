from flask import Flask, request, jsonify
from sklearn.externals import joblib
import pandas as pd
from sklearn import preprocessing
from sklearn import manifold
from sklearn.manifold import Isomap

app = Flask(__name__)

def Process(df):
	df.dropna(axis=0,how='any',inplace=True)
	df = pd.get_dummies(df,['Sector','SubSector'])
	T = preprocessing.KernelCenterer().fit_transform(df.as_matrix())
	model = Isomap(n_neighbors=5, n_components=2)
	X_test = model.fit_transform(T)
	return X_test

@app.route('/predict', methods=['GET', 'POST'])
def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_)
     query = Process(query_df)
     prediction = clf.predict(query)
     return jsonify({'prediction': prediction.tolist()})

@app.route('/test', methods=['GET'])
def test():
     return jsonify("Hello World!")

if __name__ == '__main__':
     clf = joblib.load('C:/My/pyprojects/Models/model.pkl')
     app.run(port=7000)