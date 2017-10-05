from flask import Flask  
from flask_restful import Resource, Api  
from flask_restful import reqparse  
from utils import makeprediction

app = Flask(__name__)  
api = Api(app)

class GetData(Resource):  
    def post(self):

        parser = reqparse.RequestParser()
        parser.add_argument('id')
        args = parser.parse_args()
        print (args)
        ip_id = args['id']        

        prediction_class = makeprediction.get_feature(ip_id)
        # print (type(prediction_class))

        print ("Class is : " + str(prediction_class['value']))

        return prediction_class

api.add_resource(GetData, '/getdata')

if __name__ == '__main__':  
    app.run(port=5001,debug=False)
