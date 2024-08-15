from flask import render_template,Flask,url_for
import joblib
from skimage.feature import hog
from skimage.transform import resize
import werkzeug

# fetching the image and storing it in upload folder
def fetch():
    return image

# preprocessing the image
def preprocess(image):
    image = resize(image, (200,200))
    hogify = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16.16), visualize=False, feature_vector=True)
    return hogify

# predicting the model
def predict(hogify):
    model=joblib.load(open(r'D:\IDEs\anaconda\envs\rice-disease\models\model.pkl','rb'))
    prediction=model.predict(hogify)
    return prediction

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')