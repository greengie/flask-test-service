from flask import Flask
from flask import Response  
from flask_restful import Resource, Api  
from flask_restful import reqparse
from werkzeug.exceptions import NotFound, ServiceUnavailable
from utils import decision_tree
import requests 
import pandas as pd
import pickle 

app = Flask(__name__)  
api = Api(app)

class ModelA(Resource):
	def get(self):
		print("Training Model")
		result = {"result": "model trained"}
		modelname = './model/model_a.sav'
		decision_tree.train_model('/home/thanathip/platform/setup-hadoop/training.csv', modelname)
		return result, 200

	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('id')
		args = parser.parse_args()
		# print(args)
		ip_id = args['id']

		try:    
			msg_body = requests.post('http://127.0.0.1:5001/api/getdata', data = {"id": ip_id})
		except requests.exceptions.ConnectionError:
			raise ServiceUnavailable("connection not available.")

		if msg_body.status_code == 404:
			raise NotFound("Cannot Predict for {}".format(ip_id))

		print(msg_body.json())

		X_test = pd.DataFrame.from_dict(msg_body.json(), orient='index').T
		feature_list = ['nr_employed', 'pdays', 'euribor3m', 'cons_conf_idx', 'age', 'month_oct', 'contact_telephone', 'campaign', 'poutcome_failure', 'cons_price_idx', 'emp_var_rate', 'day_of_week_mon', 'education_university.degree', 'month_apr', 'day_of_week_wed']
		X = pd.DataFrame()
		for feature in feature_list:
			try:
				X[feature] = X_test[feature]
			except:
				X[feature] = 0
		filename = './model/model_a.sav'
		model = pickle.load(open(filename, 'rb'))
		pred = model.predict(X).tolist()
		print(pred[0])
		result = {"class": pred[0]}
		return result, 200

api.add_resource(ModelA, '/api/model_a')

if __name__ == '__main__':  
    app.run(port=5000,debug=True)
