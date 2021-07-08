# import zipfile
# from google.colab import drive
# drive.mount('/content/gdrive')
# !pip install flask-ngrok
# from flask_ngrok import run_with_ngrok
# creating an instance of a flask web application

# @app.route('/<name>', methods=['GET'])
# def user(name):
#     return f"hello {name}!"

# @app.route('/admin/')
# def admin():
#     # if a: #you can add tests
#     return redirect(url_for("user", name="awatef"))


@app.route('/uploadDataset', methods=['POST'])
def upload_dataset():
    pass


#Define the route "how to access a specific page"
@app.route('/', methods=['GET'])
def home_page():
    return render_template('index.html')


UPLOAD_FOLDER = 'D:/Developement/PFE-APP/uploads/'

# creating an instance of a flask web application
app = Flask(__name__)
UPLOAD_FOLDER = 'D:/Developement/PFE-APP/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

QUERY_IMAGE_UPLOAD_FOLDER = 'uploads/QueryImage'
DATASET_UPLOAD_FOLDER = 'D:/Developement/PFE-APP/Dataset/'

#Running the app
if __name__ == "__main__":
    app.run(port=3000, debug=True)
    app.run(host='127.0.0.1', port=5000, debug=False)


@app.route('/')
def upload_form():
    return render_template('upload.html')


app.secret_key = "secret key"

# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

# # Make directory if uploads is not exists
# if not os.path.isdir(UPLOAD_FOLDER):
#     os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Allowed extension you can set your own
# ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit(
#         '.', 1)[1].lower() in ALLOWED_EXTENSIONS

from uploadFilesClass import UploadFilesClass

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         flash('No file parts')
#         print("sdsd")
#         return redirect(request.url)
#     print("ssss")
#     test = UploadFilesClass(app.config['UPLOAD_FOLDER'], request.files)
#     print("aAS")
#     test.upload()
#     return render_template('upload.html')

#############################

if __name__ == '__main__':

    # Find if any accelerator is presented, if yes switch device to use CUDA or else use CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # preparing intermediate DataFrame
    datasetPath = Path('D:/Developement/PFE-APP/static/uploads/Dataset/')
    df = pd.DataFrame()

    myfilename = []
    for dirname, _, filenames in os.walk(datasetPath):
        for filename in filenames:
            #print(os.path.join(dirname, filename))
            myfilename.append(os.path.join(dirname, filename))
    df['image'] = myfilename

    #df['image'] = [f for f in os.listdir(datasetPath) if os.path.isfile(os.path.join(datasetPath, f))]
    #df['image'] = '/content/gdrive/MyDrive/Medical MNIST/test/' + df['image'].astype(str)
    df.head()

    EPOCHS = 3
    NUM_BATCHES = 16  #
    RETRAIN = False

    train_set, validate_set = prepare_data(DF=df)

    dataloaders = {
        'train':
        DataLoader(train_set,
                   batch_size=NUM_BATCHES,
                   shuffle=True,
                   num_workers=1),
        'val':
        DataLoader(validate_set, batch_size=NUM_BATCHES, num_workers=1)
    }

    #To show images
    # images = next(iter(DataLoader(train_set, batch_size=NUM_BATCHES, shuffle=True, num_workers=1)))
    # helper.imshow(images[31], normalize=False)

    dataset_sizes = {'train': len(train_set), 'val': len(validate_set)}

    model = ConvAutoencoder_v2().to(device)

    criterion = nn.MSELoss()
    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3,
                                           gamma=0.1)  #this was commented

    freeze_support()

    model, optimizer, loss = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=exp_lr_scheduler,  #this was commented
        num_epochs=EPOCHS)

    # Save the Trained Model
    torch.save(
        {
            'epoch': EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        },
        '/content/gdrive/MyDrive/Expérimentations documents/conv_autoencoder_v2_Exp#1.pt'
    )

    # extractor = parallelTestModule.ParallelExtractor()
    # extractor.runInParallel(numProcesses=2, numThreads=4)
#  python model_and_training.py


##############################################
#Uploading query image
@app.route('/uploadQueryImage', methods=['POST'])
def upload_image():
    imagefile = request.files['imagefile']
    image_path = QUERY_IMAGE_UPLOAD_FOLDER + imagefile.filename
    imagefile = Utils.pre_processing(imagefile)
    imagefile.save(image_path)
    #redirect(url_for("extract_query_image_features", image_path=image_path))
    return redirect(url_for('home_page'))


@app.route('/uploadQueryImages', methods=['POST'])
def upload_image():
    imagefile = request.files['imagefile']
    image_path = QUERY_IMAGE_UPLOAD_FOLDER + imagefile.filename
    imagefile = preprocessing(imagefile)
    imagefile.save(image_path)
    return render_template('index.html')