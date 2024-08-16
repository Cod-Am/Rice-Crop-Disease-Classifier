from flask import render_template,Flask,url_for,request
import os
import joblib
from skimage.io import imread
from skimage.feature import hog
from skimage.transform import resize
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

# fetching the image from the user uploaded file and saving it in the uploads folder
def fetch(image):
    filename=secure_filename(image.filename)
    path=os.path.join(app.config['uploadfolder'], filename)
    image.save(path)
    greyscale_image=imread(path,as_gray=True)
    return greyscale_image

# preprocessing the image
def preprocess(image):
    image = resize(image, (200,200))
    hogify = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16.16), visualize=False, feature_vector=True)
    return hogify

# predicting the model
def predict(hogify):
    model=joblib.load(open(r'D:\IDEs\anaconda\envs\rice-disease\models\model.pkl','rb'))
    prediction=model.predict([hogify])
    print(type(prediction))
    print(prediction)
    return prediction

# initializing the flask app and setting up the upload folder for storing the images uploaded by the user
app=Flask(__name__,template_folder='./templates')
UPLOAD_FOLDER=r'D:\IDEs\anaconda\envs\rice-disease\\data\Rice_Diseases\uploads'
app.config['uploadfolder']=UPLOAD_FOLDER
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        if request.method == 'POST':
            image = request.files['img']
            if image:
                print('image found')
                saved_image = fetch(image)
                hogify = preprocess(saved_image)
                prediction = predict(hogify)

                # Define the mapping of predictions to descriptions
                descriptions = {
                    1: 'Leaf is diseased with Rice Bacterial Blight',
                    2: 'Leaf is diseased with Rice Blast',
                    3: 'Leaf is diseased with Rice Brown Spot',
                    4: 'Leaf is diseased with Rice Leaf Smut',
                    5: 'Leaf is diseased with Rice Tungro'
                }

                # Convert prediction to description
                converted_prediction = descriptions.get(prediction[0], 'Unknown Disease')

                print(converted_prediction)
                return render_template('result.html', prediction=converted_prediction)
            else:
                return render_template('error.html', error_message="No image file found.")
        else:
            return render_template('error.html', error_message="Invalid request method.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return render_template('error.html', error_message="An internal error occurred.")
