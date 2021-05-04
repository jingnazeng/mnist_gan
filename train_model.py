import model_condi as dcgan
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

NUM_EPOCHS=10

(X_train, Y_train), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)
X_train=tf.image.resize(X_train,(dcgan.IMG_SIZE,dcgan.IMG_SIZE)) 
BUFF_SIZE=X_train.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train)).shuffle(BUFF_SIZE).batch(dcgan.BATCH_SIZE)
dcgan.train(train_dataset, NUM_EPOCHS)
