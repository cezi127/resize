from flask import Flask, jsonify, request, Response
from flask_restful import Api, Resource
import resize

app = Flask(__name__)
api = Api(app)


class PropImage(Resource):

    def return_img(self, img_local_path):
        import base64
        with open(img_local_path, 'rb') as img_f:
            img_stream = img_f.read()
            img_stream = base64.b64encode(img_stream)
        return img_stream

    def post(self):
        json_data = request.get_json()
        print(json_data["jingwei"])
        img_path = resize.process_image(json_data)
        print(img_path)
        image = open(img_path, "rb")
        return Response(image, mimetype="image/jpeg")


api.add_resource(PropImage, '/image')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
