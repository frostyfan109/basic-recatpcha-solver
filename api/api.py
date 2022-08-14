from flask import Flask
from flask_restx import Api, Resource
from flask_cors import CORS
from werkzeug import datastructures
from captcha import recaptcha

app = Flask(__name__)
CORS(app)
api = Api(app)

recaptcha_parser = api.parser()
recaptcha_parser.add_argument("row", type=int, default=3, location="form")
recaptcha_parser.add_argument("col", type=int, default=3, location="form")
recaptcha_parser.add_argument("text", type=str, location="form")
recaptcha_parser.add_argument("file", type=datastructures.FileStorage, location="files")
@api.route("/recaptcha/image")
class ImageRecaptcha(Resource):
    @api.expect(recaptcha_parser)
    def post(self):
        args = recaptcha_parser.parse_args()
        text = args["text"]
        file = args["file"]
        row = args["row"]
        col = args["col"]

        return recaptcha.solve(text, file.read(), row, col)

class SquareRecaptcha(Resource):
    def post(self):
        ...

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Specify API arguments")
    parser.add_argument("-r", "--reloader", help="Automatically restart API upon modification", action="store_true", default=True)
    args = parser.parse_args()

    port = 8081

    app.run(
        host="0.0.0.0",
        port=port,
        use_reloader=args.reloader
    )