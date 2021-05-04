import os
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices!=[]:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

NUM_CLASSES=10
NUM_EPOCHS=100
IMG_SIZE=32
(X_train, Y_train), (_, _) = mnist.load_data()
# IMG_SIZE=X_train.shape[1]
# model = tf.keras.Sequential()
# model.add(layers.Embedding(NUM_CLASSES, IMG_SIZE*IMG_SIZE))
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)
X_train=tf.image.resize(X_train,(IMG_SIZE,IMG_SIZE)) 
BUFF_SIZE=X_train.shape[0]
BATCH_SIZE=8
LATENT_DIM=100
EMBED_DIM=100
train_dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train)).shuffle(BUFF_SIZE).batch(BATCH_SIZE)
# for image_batch,label_batch in train_dataset:
#     fig = plt.figure(figsize=(4,4))
#     for i in range(8):
#         plt.subplot(4, 4, i+1)
#         plt.imshow(image_batch[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
#         plt.axis('off')

#     image_batch=tf.image.resize(image_batch,(IMG_SIZE,IMG_SIZE))
    
#     for i in range(8):
#         plt.subplot(4, 4, i+9)
#         plt.imshow(image_batch[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
#         plt.axis('off')
#     plt.show()


def generator():
    noise_inut=keras.Input(shape=(1,1,LATENT_DIM),name='img')
    target_digit=keras.Input(shape=(1,1),name='target')
    target_feature=layers.Embedding(NUM_CLASSES,EMBED_DIM)(target_digit)
    inter_layer=layers.Concatenate()([noise_inut,target_feature])
    inter_layer=layers.Dense(4*4*1024)(inter_layer)
    inter_layer=layers.Reshape((4,4,1024))(inter_layer)
    # inter_layer=layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same")(inter_layer)
    # inter_layer=layers.BatchNormalization()(inter_layer)
    # inter_layer=layers.ReLU()(inter_layer)
    inter_layer=layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")(inter_layer)
    inter_layer=layers.BatchNormalization()(inter_layer)
    inter_layer=layers.ReLU()(inter_layer)
    inter_layer=layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(inter_layer)
    inter_layer=layers.BatchNormalization()(inter_layer)
    inter_layer=layers.ReLU()(inter_layer)
    out_layer=layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding="same",activation="tanh")(inter_layer)
    return keras.Model(inputs=[noise_inut,target_digit],outputs=out_layer,)

gen=generator()
gen.summary()
keras.utils.plot_model(gen, "DCGAN_gen_model.png", show_shapes=True)

def discriminator():
    img_inut=keras.Input(shape=(IMG_SIZE,IMG_SIZE,1))
    target_digit=keras.Input(shape=(1,1),name='target')
    target_feature=layers.Embedding(NUM_CLASSES,IMG_SIZE*IMG_SIZE)(target_digit)
    target_feature=layers.Reshape((IMG_SIZE,IMG_SIZE,1))(target_feature)
    inter_layer=layers.Concatenate()([img_inut,target_feature])
    # print(inter_layer.shape)
    inter_layer=layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(inter_layer)
    inter_layer=layers.BatchNormalization()(inter_layer)
    inter_layer=layers.LeakyReLU(0.2)(inter_layer)
    inter_layer=layers.Conv2D(256, kernel_size=4, strides=2, padding="same")(inter_layer)
    inter_layer=layers.BatchNormalization()(inter_layer)
    inter_layer=layers.LeakyReLU(0.2)(inter_layer)
    inter_layer=layers.Conv2D(512, kernel_size=4, strides=2, padding="same")(inter_layer)
    inter_layer=layers.BatchNormalization()(inter_layer)
    inter_layer=layers.LeakyReLU(0.2)(inter_layer)
    # inter_layer=layers.Conv2D(1024, kernel_size=4, strides=2, padding="same")(inter_layer)
    # inter_layer=layers.BatchNormalization()(inter_layer)
    # inter_layer=layers.LeakyReLU(0.2)(inter_layer)
    inter_layer=layers.Flatten()(inter_layer)
    out_layer=layers.Dense(1, activation="sigmoid")(inter_layer)
    return keras.Model(inputs=[img_inut,target_digit],outputs=out_layer,)

dis=discriminator()
dis.summary()
keras.utils.plot_model(dis, "DCGAN_dis_model.png", show_shapes=True)
#

opt_gen = keras.optimizers.Adam(1e-4)
opt_disc = keras.optimizers.Adam(1e-4)
# loss_fn = keras.losses.BinaryCrossentropy()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = (real_loss + fake_loss)/2
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(opt_gen=opt_gen,
                                 opt_disc=opt_disc,
                                 gen=gen,
                                 dis=dis)


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch,label_batch in dataset:
            # train_step(image_batch)
            noise_vec=tf.random.normal(shape=(BATCH_SIZE,1,1, LATENT_DIM))
            label_batch=tf.reshape(label_batch,[BATCH_SIZE,1,1])
            # fake_imgs=gen([noise_vec,label_batch],training=True)
            with tf.GradientTape() as disc_tape,tf.GradientTape() as gen_tape:
                generated_images = gen([noise_vec,label_batch], training=True)
                real_output = dis([image_batch,label_batch], training=True)
                fake_output = dis([generated_images,label_batch], training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)
            
            gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, dis.trainable_variables)

            opt_gen.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
            opt_disc.apply_gradients(zip(gradients_of_discriminator, dis.trainable_variables))

   # Save the model every 15 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            test_noise_vec=tf.random.normal(shape=(10,1,1, LATENT_DIM))
            test_label=tf.reshape(tf.constant([0,1,2,3,4,5,6,7,8,9]),[10,1,1])
            test_result=gen([test_noise_vec,test_label])
            fig = plt.figure(figsize=(2,5))
            for i in range(test_result.shape[0]):
                plt.subplot(2, 5, i+1)
                plt.imshow(test_result[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
                plt.axis('off')

            plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

 # Generate after the final epoch
    # display.clear_output(wait=True)
    # generate_and_save_images(generator,epochs,seed)
train(train_dataset, NUM_EPOCHS)

gen.save('DCGAN_condi_gen.h5')
