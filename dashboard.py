from myImports import *
from UploadFiles import UploadFiles

# creating an instance of a flask web application
app = Flask(__name__, template_folder='templates', static_folder='static')
# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')
# Make directory if "uploads" folder not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['csv', 'png', 'jpg', 'jpeg', 'zip'])
app.secret_key = "secret key"


#Define the route "how to access a specific page"
@app.route('/', methods=['GET'])
def dashboard_page():
    return render_template('dashboard.html')


@app.route('/trainModelPage', methods=['GET'])
def train_model_page():
    return render_template('train_model.html')


@app.route('/searchPage', methods=['GET'])
def search_page():
    return render_template('train_model.html')


@app.route('/upload/<path>/<filename>')
def download_file(path, filename):
    return send_from_directory(path, filename, as_attachment=True)


@app.route('/upload', methods=['POST'])
def upload_Train_set():
    if 'file' not in request.files:
        flash('No file parts')
        print("sdsd")
        return redirect(request.url)
    print("ssss", app.config['UPLOAD_FOLDER'])
    train_set = UploadFiles(app.config['UPLOAD_FOLDER'], request.files)
    print("aAS")
    train_set_path = train_set.upload()
    print(train_set_path)
    os.path.exists(os.path.abspath(train_set_path))
    print('ss', train_set_path)

    session['train_set_path'] = train_set_path

    images_to_display = []
    counter = 0
    for dirname, _, filenames in os.walk(train_set_path):
        for filename in filenames:
            if counter == 8:
                break
            temp_list = []
            temp_list.append(dirname)
            temp_list.append(filename)
            # imagePath = os.path.join(dirname, filename)
            images_to_display.append(temp_list)
            counter = counter + 1

    print(images_to_display)
    return render_template('train_model.html',
                           images_to_display=images_to_display)
    #return redirect(url_for('train_and_save'))


#Running the app
if __name__ == "__main__":
    app.run(port=3000, debug=True)