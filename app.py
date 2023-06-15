from flask import Flask, jsonify, request
from static.inference import inference_model, transform_data

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


@app.route('/root', methods=["POST", "GET"])
def root():
    return {"message": "Hello World!"}


@app.route('/predict', methods=["POST"])
def predict():  # put application's code here
    if request.method == "POST":
        file = request.files['file']
        img_bytes = file.read()
        tensor = transform_data(img_bytes)
        outputs = inference_model(tensor)

        return jsonify({"class_is": outputs})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="5000")
