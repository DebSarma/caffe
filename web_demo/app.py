import cStringIO as StringIO
import datetime
import logging
import optparse
import os
import time
import urllib
import caffe
import flask
import numpy as np
import tornado.httpserver
import tornado.wsgi
import werkzeug
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import exifutil

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/..')
UPLOAD_FOLDER = '/tmp/object_detection_demo_uploads'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'}

# load PASCAL VOC labels
_labelmap_file = '{}/data/VOC0712/labelmap_voc.prototxt'.format(ROOT_DIR)
_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(open(_labelmap_file, 'r').read()), _labelmap)


def get_labelname(labels):
    num_labels = len(_labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == _labelmap.item[i].label:
                found = True
                labelnames.append(_labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/detect_url', methods=['GET'])
def detect_url():
    image_url = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(image_url).read())
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )
    logging.info('Image: %s', image_url)
    result = app.detector.detect(image)
    return flask.render_template('index.html', has_result=True, result=result)


@app.route('/detect_upload', methods=['POST'])
def detect_upload():
    try:
        # We will save the file to disk for possible data collection.
        image_file = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
                    werkzeug.secure_filename(image_file.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        image_file.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    result = app.detector.detect(image)
    return flask.render_template('index.html', has_result=True, result=result)


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


prototxt = '{}/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'.format(ROOT_DIR)
caffemodel = '{}/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'.format(ROOT_DIR)

confidence_threshold = 0.6

colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()


def _get_image_with_detections(image, detections):
    # Parse the outputs.
    det_label = detections[0, 0, :, 1]
    det_conf = detections[0, 0, :, 2]
    det_xmin = detections[0, 0, :, 3]
    det_ymin = detections[0, 0, :, 4]
    det_xmax = detections[0, 0, :, 5]
    det_ymax = detections[0, 0, :, 6]

    # Get detections with confidence higher than threshold
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= confidence_threshold]
    top_conf = det_conf[top_indices]
    top_labels = det_label[top_indices].tolist()
    top_label_names = get_labelname(top_labels)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    # Render image with detections
    plt.clf()
    plt.axis('off')
    plt.imshow(image)
    current_axis = plt.gca()

    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = int(top_labels[i])
        label_name = top_label_names[i]
        print xmin, ymin, xmax, ymax, score, label_name
        display_txt = '%s: %.2f' % (label_name, score)
        coordinates = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        color = colors[label]
        current_axis.add_patch(plt.Rectangle(*coordinates, fill=False, edgecolor=color, linewidth=2))
        current_axis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

    result = StringIO.StringIO()
    plt.savefig(result, format='png', bbox_inches='tight')
    result.seek(0)
    return 'data:image/png;base64,' + result.getvalue().encode('base64').replace('\n', '')


class ObjectDetector(object):
    def __init__(self, prototxt, caffemodel, gpu_id):
        logging.info('Loading net and associated files...')
        if gpu_id < 0:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(gpu_id)
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        image_resize = 300
        self.net.blobs['data'].reshape(1, 3, image_resize, image_resize)
        logging.info('Loaded network {:s}'.format(caffemodel))

        # input pre-processing: 'data' is the name of the input blob == net.inputs[0]
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        transformer.set_raw_scale('data',
                                  255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
        self.transformer = transformer

    def detect(self, image):
        try:
            start = time.time()
            transformed_image = self.transformer.preprocess('data', image)
            self.net.blobs['data'].data[...] = transformed_image

            # Forward pass.
            detections = self.net.forward()['detection_out']
            elapsed = time.time() - start
            return True, _get_image_with_detections(image, detections), elapsed

        except Exception as err:
            logging.info('Detection error: %s', err)
            return False, 'Something went wrong when detecting the image. Maybe try another one?'


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option('-g', '--gpu',
                      type=int, default=0,
                      help="GPU id; -1 if using CPU")

    opts, args = parser.parse_args()
    # Initialize classifier + warm start by forward for allocation
    app.detector = ObjectDetector(prototxt, caffemodel, opts.gpu)
    app.detector.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
