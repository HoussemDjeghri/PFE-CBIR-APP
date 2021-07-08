from myImports import *
from UploadFiles import UploadFiles

# creating an instance of a flask web application
app = Flask(__name__)
app.static_folder = 'static'

UPLOAD_FOLDER = 'D:/Developement/PFE-APP/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

QUERY_IMAGE_UPLOAD_FOLDER = 'D:/Developement/PFE-APP/static/uploads/QueryImage'
DATASET_UPLOAD_FOLDER = 'D:/Developement/PFE-APP/static/uploads/Dataset'


#Define the route "how to access a specific page"
@app.route('/', methods=['GET'])
def home_page():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_Train_set():
    if 'file' not in request.files:
        flash('No file parts')
        print("sdsd")
        return redirect(request.url)
    print("ssss")
    train_set = UploadFiles(app.config['UPLOAD_FOLDER'] + 'Dataset/train',
                            request.files)
    print("aAS")
    Train_set_path = train_set.upload()
    return render_template('upload.html')


# #Uploading query image
# @app.route('/uploadQueryImage', methods=['POST'])
# def upload_image():
#     imagefile = request.files['imagefile']
#     image_path = QUERY_IMAGE_UPLOAD_FOLDER + imagefile.filename
#     imagefile = preprocessing(imagefile)
#     imagefile.save(image_path)
#     #redirect(url_for("extract_query_image_features", image_path=image_path))
#     return redirect(url_for('home_page'))

# #Extracting query image features
# @app.route('/extractImageFeatures', methods=['POST'])
# def extract_query_image_features():
#     #TODO
#     return render_template('index.html')


#Preprocessing image
def preprocessing(image_path):
    img = Image.open(image_path)
    x, y = img.size
    size = max(512, x, x)
    new_im = Image.new('RGB', (size, size), (0, 0, 0))
    new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


#Running the app
#Running the app
if __name__ == "__main__":
    app.run(port=3000, debug=True)