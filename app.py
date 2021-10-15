from flask import Flask, render_template, request, flash, redirect
from models import MobileNet
import os
import pandas as pd

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
#It will allow below 16MB contents only, you can change it
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'templates/uploads')

# Make directory if "uploads" folder not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = MobileNet()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


# @app.route('/infer', methods=['POST'])
# def success():
#     if request.method == 'POST':
#         f = request.files.getlist("images")
#         print('3#$@#$#@$', f)
#         saveLocation = f.filename
#         print('$#%%^$%@#$#@', saveLocation)
#         f.save(saveLocation)
#         # inference, confidence = model.infer(saveLocation)
#         infer_dict = model.infer(saveLocation)
#         print(infer_dict)
#         inference, confidence = infer_dict[0]
#         # make a percentage with 2 decimal points
#         confidence = floor(confidence * 10000) / 100
#         # delete file after making an inference
#         os.remove(saveLocation)
#         # respond with the inference
#         return render_template('inference.html', name=inference, confidence=confidence)



@app.route('/infer', methods=['POST'])
def success():
    if request.method == 'POST':

        print(f'DSFASD {request.files.listvalues()}')

        # file1 = request.files.to_dict()['file']
        # file2 = request.files[1]
        # uploaded_files = request.files.getlist("file")
        # print('Uplaoded files are', uploaded_files)
        # print(f'#$@#$@$ files: {file1[0].filename}')
        
        saveLocationList = []

        for file in list(request.files.getlist("file")):
            if file:
                print(f'%$$%#$ Individual File name :  {file}')
                saveLocation = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(saveLocation)
                saveLocationList.append(saveLocation)
                print(saveLocation)

        infer_dict = model.infer(saveLocationList)
        df_pred = pd.read_csv('predictions.csv')
        df_pred1 = df_pred

        for predictions in infer_dict.values():
            print(list(predictions))
            new_entry = {'path': predictions[2], 'prediction': predictions[0], 'accuracy':predictions[1]}
            df_pred.loc[len(df_pred)] = new_entry
        
        df_pred.to_csv('predictions.csv', index=False)

        print(infer_dict)
        # make a percentage with 2 decimal points
        # confidence = floor(confidence * 10000) / 100
        # delete file after making an inference
        # os.remove(saveLocation)

        # flash('File(s) successfully uploaded')
        return render_template('inference.html',dict_results=infer_dict, tables=[df_pred1.to_html(classes='data', header="true")])




if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port, debug=True)
