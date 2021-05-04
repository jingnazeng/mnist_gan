from flask import Flask, request, render_template
import os
from PIL import Image
import base64
import io
import tensorflow as tf
from tensorflow import keras
from model_condi import  gen
from matplotlib import pyplot
from numpy.random import randn
import numpy as np

app = Flask(__name__)

file_path=os.path.dirname(os.path.realpath(__file__))

Z_DIM=100
target_digit= 4
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NUM_CLASSES=10
GEN_EMBEDDING=100
FEATURES_GEN = 16

gen=keras.models.load_model('DCGAN_condi_gen_best2.h5')

def generate_mnist(target_digit=0):
    target_digit=tf.reshape(target_digit,[1,1,1])
    noise_vec=tf.random.normal(shape=(1,1,1, Z_DIM),seed=1)
    X=gen([noise_vec,target_digit])
    im_plt = pyplot.imshow(tf.squeeze(X), cmap='gray_r')
    im = Image.fromarray(np.uint8(im_plt.get_cmap()(im_plt.get_array())*255))
    im = im.resize((64,64))
    data = io.BytesIO()
    rgb_im=im.convert('L')
    rgb_im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return encoded_img_data

@app.route("/")
def index():
    return render_template('index.html',img_data=generate_mnist(0).decode('utf-8'),error=[])


@app.route("/generate", methods=['POST'])
def generate():
    target_digit = request.form['text']
    # assert 0<=target_digit<=9
    if target_digit.isdigit():
        if 0 <= int(target_digit) <= 9:
            return render_template('index.html',img_data=generate_mnist(int(target_digit)).decode('utf-8'),error=[])
        else:
            return render_template('index.html',img_data=[],error='Please give an integer between 0~9!')         
    else:
        return render_template('index.html',img_data=[],error='Please give an integer between 0~9!')

if __name__ == "__main__":
    app.run(debug='True')