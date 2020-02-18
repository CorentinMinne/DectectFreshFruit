from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


from load import *

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

model, optimizer = load_checkpoint(path=checkpoint_path)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index_view():
    return render_template('index.html')


def convertImage(req):
    if 'file' not in req.files:
        print("No file")
        return None
    file = req.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        print("No file")
        return None
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join("./", filename))
        return os.path.join("./", filename)
    return None


@app.route('/predict/', methods=['GET','POST'])
def predict():
    # imgData = request.get_data()
    url = convertImage(request)
    img, ps, classes, y_obs = predict_img(url, model, n_classes)
    os.remove(url)
    return classes[np.argmax(ps)]


if __name__ == '__main__':
    app.run(debug=True)
