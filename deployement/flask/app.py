import numpy as np
import pickle
from werkzeug.utils import secure_filename
import os
from skimage.feature import hog
from skimage.transform import resize
from skimage.io import imread
from flask import render_template, Flask, request, flash

app = Flask(__name__, template_folder='../templates')
UPLOAD_FOLDER = 'D:/IDEs/anaconda/envs/rice-disease/data/deployement/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_image(image_path):
    greyscale_image = imread(image_path, as_gray=True, plugin='pil')
    resized_greyscale_image = resize(greyscale_image, (200, 200))
    hogify, hog_img = hog(
        resized_greyscale_image, orientations=8,
        pixels_per_cell=(4, 4),
        cells_per_block=(8, 8),
        visualize=True
    )
    model_path = r'D:/IDEs/anaconda/envs/rice-disease/model/model_100.pkl'
    model = pickle.load(open(model_path, 'rb'))
    prediction = model.predict(hogify.reshape(1, -1))  # Ensure input shape matches model's expectation
    return prediction

@app.route('/home')
def home():
    return render_template('form.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        uploaded_image = request.files.get('fileUpload')
        if uploaded_image:
            img_filename = secure_filename(uploaded_image.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
            uploaded_image.save(img_path)
            prediction_number = predict_image(img_path)
            prediction_map = {
                1: 'Bacterial Blight Disease',
                2: 'Blast Disease',
                3: 'Brown Spot Disease',
                4: 'False Smut Disease'
            }
            prediction_text = prediction_map.get(prediction_number[0], 'Unknown Disease')
            return render_template('result.html', prediction=prediction_text)
        else:
            return render_template('result.html', prediction='No image found')

if __name__ == '__main__':
    app.run(debug=True)
