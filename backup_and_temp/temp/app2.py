import os
import uuid
import shutil
import zipfile
from flask import Flask, render_template, request, redirect, url_for
# from werkzeug.datastructures import FileStorage

app = Flask(__name__)

UPLOAD_FOLDER = 'D:/Developement/PFE-APP/static/uploads/Dataset'

# BASE_DIR = os.path.dirname(os.path.abspath(__file__), "/uploads")
BASE_DIR = UPLOAD_FOLDER


def unzip_file(zip_src, dst_dir):
    """
    Unzip the zip file
         :param zip_src: full path of zip file
         :param dst_dir: the destination folder to extract to
    :return:
    """
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, "r")
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        return "Please upload zip file"


#Define the route "how to access a specific page"
@app.route('/tete', methods=['GET'])
def home_page():
    return render_template('upload.html', content=["haha", "hoho", "hihi"])


@app.route("/upload", methods=["GET", "POST"])
def upload():
    # if request.method == "GET":
    #     return render_template("upload.html")
    obj = request.files.get("file")
    # print(obj)  # <FileStorage: "test.zip" ("application/x-zip-compressed")>
    # print(obj.filename)  # test.zip
    # print(obj.stream)  # <tempfile.SpooledTemporaryFile object at 0x0000000004135160>
    #      # Check if the suffix name of the uploaded file is zip
    ret_list = obj.filename.rsplit(".", maxsplit=1)
    if len(ret_list) != 2:
        return "Please upload zip file"
    if ret_list[1] != "zip":
        return "Please upload zip file"

    # Method 1: Save the file directly
    # obj.save(os.path.join(BASE_DIR, "files", obj.filename))

    # Method 2: Save the decompressed file (the original compressed file is not saved)
    # target_path = os.path.join(BASE_DIR, "files", str(uuid.uuid4()))
    # shutil._unpack_zipfile(obj.stream, target_path)

    # Method three: Save the compressed file locally, then decompress it, and then delete the compressed file
    file_path = os.path.join(
        BASE_DIR, obj.filename)  # The path where the uploaded file is saved
    obj.save(file_path)
    target_path = os.path.join(BASE_DIR, str(
        uuid.uuid4()))  # The path where the unzipped files are saved
    ret = unzip_file(file_path, target_path)
    os.remove(file_path)  # delete file
    if ret:
        return ret

    return render_template('upload.html')


if __name__ == "__main__":
    app.run(debug=True)