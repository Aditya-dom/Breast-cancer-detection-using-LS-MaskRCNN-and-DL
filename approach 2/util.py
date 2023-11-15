import os
from PIL import Image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def convertToJpg(app, filename):
    image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    if '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg'}:  # no need for conversion
        return

    new_file_name = filename.rsplit('.', 1)[0]+'.jpg'
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], new_file_name))

    # deleting previous file
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # returning new file path
    return os.path.join(app.config['UPLOAD_FOLDER'], new_file_name)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
