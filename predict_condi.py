import os
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot
# from model_condi import opt_gen,opt_disc,gen,dis,checkpoint,checkpoint_dir
from model_condi import gen

LATENT_DIM=100


gen=keras.models.load_model('DCGAN_condi_gen_best2.h5')
noise_vec=tf.random.normal(shape=(1,1,1, LATENT_DIM))
# print(noise_vec)
target_digit=2
assert 0<=target_digit<=10
target_digit=tf.constant([target_digit])
target_digit=tf.reshape(target_digit,[1,1,1])
X=gen([noise_vec,target_digit])
# gen.save('DCGAN_condi_gen.h5')
X=tf.squeeze(X)
pyplot.imshow(tf.squeeze(X), cmap='gray_r')
pyplot.show()
