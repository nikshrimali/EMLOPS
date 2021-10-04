from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
from torch_utils import transform_image, get_prediction
from PIL import Image
from base64 import b64encode
import io

#MYDIR = os.path.dirname(__file__)
UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    if request.method == 'POST':
        print('post request received')

        img_file = request.files.get('file')

        if img_file is None or img_file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(img_file.filename):
            return jsonify({'error': 'format not supported'})

        # img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename))

        try:
            img_bytes = img_file.read()

            tensor = transform_image(img_bytes)
            # img_bytes.save(os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename))

            prediction = get_prediction(tensor)

            data = {'prediction': prediction}
            # image = b64encode(img_bytes.image).decode("utf-8")
            image = Image.open(io.BytesIO(img_bytes))
            path=app.config['UPLOAD_FOLDER']
            os.makedirs(path)
            image.save(os.path.join(path, img_file.filename))
            return render_template('main.html', value=prediction, image=img_file.filename)
            return jsonify(data)
        except Exception as e:
            print(e)
            return jsonify({'error': 'error during prediction'})


if __name__ == '__main__':
    app.run(debug=True)