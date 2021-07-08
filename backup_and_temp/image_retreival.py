from logging import debug
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import os

# creating an instance of a flask web application
app = Flask(__name__)
app.static_folder = 'static'

# UPLOAD_FOLDER = 'D:/Developement/PFE-APP/uploads/'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

QUERY_IMAGE_UPLOAD_FOLDER = 'D:/Developement/PFE-APP/static/uploads/QueryImage'
DATASET_UPLOAD_FOLDER = 'D:/Developement/PFE-APP/static/uploads/Dataset'


#Define the route "how to access a specific page"
@app.route('/', methods=['GET'])
def home_page():
    return render_template('index.html')


#Uploading query image
@app.route('/uploadQueryImage', methods=['POST'])
def upload_image():
    imagefile = request.files['imagefile']
    image_path = QUERY_IMAGE_UPLOAD_FOLDER + imagefile.filename
    imagefile = preprocessing(imagefile)
    imagefile.save(image_path)
    #redirect(url_for("extract_query_image_features", image_path=image_path))
    return redirect(url_for('home_page'))


#Extracting query image features
@app.route('/extractImageFeatures', methods=['POST'])
def extract_query_image_features():
    #TODO
    return render_template('index.html')


#Preprocessing image
def preprocessing(image_path):
    img = Image.open(image_path)
    x, y = img.size
    size = max(512, x, x)
    new_im = Image.new('RGB', (size, size), (0, 0, 0))
    new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


#Running the app
if __name__ == "__main__":
    app.run(port=3000, debug=True)
