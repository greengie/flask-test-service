from flask import Flask
from flask import Response  
from flask_restful import Resource, Api  
from flask_restful import reqparse
from werkzeug.exceptions import NotFound, ServiceUnavailable
import requests  

app = Flask(__name__)  
api = Api(app)

class ModelA(Resource):
	def get(self):
		print("in model a")
		return "get", 200

	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('id')
		args = parser.parse_args()
		# print(args)
		ip_id = args['id']

		try:    
			pred_class = requests.post('http://127.0.0.1:5001/api/getdata', data = {"id": ip_id})
		except requests.exceptions.ConnectionError:
			raise ServiceUnavailable("connection not available.")

		if pred_class.status_code == 404:
			raise NotFound("Cannot Predict for {}".format(ip_id))

		print(pred_class.json())
		return 'post', 200


api.add_resource(ModelA, '/api/model_a')

if __name__ == '__main__':  
    app.run(port=5000,debug=True)
