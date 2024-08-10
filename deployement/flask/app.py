import numpy
import compress_pickle

from skimage.feature import hog
from skimage.transform import resize
from skimage.io import imread

from flask import render_template,Flask,request

def predict_image(image):
    greyscale_image=imread(image,as_gray=True)
    resized_greyscale_image=resize(greyscale_image,(200,200))
    hogify, hog_img = hog(
    resized_greyscale_image, orientations=8,
    pixels_per_cell=(4, 4),
    cells_per_block=(8,8),
    visualize=True
    )
    model=compress_pickle.load('D:\IDEs/anaconda\envs/rice-disease\model\model_100.pkl')
    prediction=model.predict(hogify)
    return prediction

app=Flask(__name__,template_folder='../templates')
@app.route('/home')
def home():
    return render_template('form.html')

@app.route('/result',methods=['POST'])
def result():
    if request.method == 'POST':
        image=request.form.get('image')
        prediction_number=predict_image(image)
        if prediction_number == 1:
            return render_template('result.html',prediction='Becterial Blight Disease')
        if prediction_number == 2:
            return render_template('result.html',prediction='Blast Disease')

        if prediction_number == 3:
            return render_template('result.html',prediction='Brown Spot Disease')

        if prediction_number == 4:
            return render_template('result.html',prediction='False Smut Disease')

