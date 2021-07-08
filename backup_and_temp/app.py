from logging import debug
from flask import Flask, render_template, request, redirect, url_for
# from werkzeug.utils import secure_filename
# from werkzeug.datastructures import  FileStorage
# from flask_uploads import UploadSet, configure_uploads, IMAGES
from PIL import Image
import os

UPLOAD_FOLDER = 'D:/Developement/PFE-APP/uploads/'

# creating an instance of a flask web application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

QUERY_IMAGE_UPLOAD_FOLDER = 'D:/Developement/PFE-APP/QueryImage/'
DATASET_UPLOAD_FOLDER = 'D:/Developement/PFE-APP/Dataset/'

#Define the route "how to access a specific page"
@app.route('/', methods=['GET']) 
def home_page(): 
    return render_template('index.html', content = ["haha","hoho", "hihi"])

@app.route('/uploadQueryImages', methods=['POST'])
def upload_image():
    imagefile = request.files['imagefile']
    image_path = QUERY_IMAGE_UPLOAD_FOLDER + imagefile.filename
    imagefile = preprocessing(imagefile)
    imagefile.save(image_path)
    return render_template('index.html', content = ["haha","hoho", "hihi"])

@app.route('/uploadDataset', methods=['POST'])
def upload_dataset():
    datasetfolder = request.files.path['datasetfolder[]']
    datasetfolder_path = os.path.join(app.config['UPLOAD_FOLDER'] +'/', datasetfolder.filename)
    datasetfolder.save(datasetfolder_path)

    return render_template('index.html',content = ["haha","hoho", "hihi"])




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




# @app.route('/<name>', methods=['GET']) 
# def user(name): 
#     return f"hello {name}!"

# @app.route('/admin/') 
# def admin(): 
#     # if a: #you can add tests
#     return redirect(url_for("user", name="awatef"))