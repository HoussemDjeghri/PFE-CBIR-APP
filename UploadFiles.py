from myImports import *


class UploadFiles:

    ALLOWED_EXTENSIONS = set(['zip', 'csv', 'png', 'jpg', 'jpeg'])

    def __init__(self, upload_path, request_files):
        self.upload_path = upload_path
        self.request_files = request_files

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit(
            '.', 1)[1].lower() in self.ALLOWED_EXTENSIONS

    # def upload_file(self):
    #     if 'files[]' not in self.request_files:
    #         flash('No file part')
    #         return redirect(request.url)

    #     files = self.request_files.getlist('files[]')

    #     for file in files:
    #         if file and self.allowed_file(file.filename):
    #             filename = secure_filename(file.filename)
    #             file.save(os.path.join(self.upload_path, filename))

    #     flash('File(s) successfully uploaded')

    def unzip_file(self, zip_src, dst_dir):
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

    def upload(self,nameOfFile):
        fiiiileName=nameOfFile
        if not nameOfFile:
            fiiiileName = 'file'
        obj = self.request_files.get(fiiiileName)
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

        #Method 2: Save the decompressed file (the original compressed file is not saved)
        # target_path = os.path.join(self.upload_path, "files",
        #                            str(uuid.uuid4()))
        # shutil._unpack_zipfile(obj.stream, target_path)

        # # Method three: Save the compressed file locally, then decompress it, and then delete the compressed file
        file_path = os.path.join(
            self.upload_path,
            obj.filename)  # The path where the uploaded file is saved
        obj.save(file_path)
        target_path = os.path.join(self.upload_path, str(
            uuid.uuid4()))  # The path where the unzipped files are saved
        ret = self.unzip_file(file_path, target_path)
        print(target_path)
        os.remove(file_path)  # delete file
        # if ret:
        #     return ret
        return target_path