from keras.datasets import mnist

import numpy as np
import time
from os import listdir
from PIL import Image
from scipy import misc, ndimage
import keras.backend as K
#import Image
#from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.datasets import cifar10, mnist
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Convolution2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

data_path = "/work/04381/ymarathe/maverick/dogscats/train/cats/"





img_rows, img_cols = 28, 28
num_images = 10000
channel = 3

imagesList = listdir(data_path)
loadedImages = []
i = 0
for image in imagesList:
    i += 1
    print i
    if i > num_images:
        break
    #img = Image.open(data_path + image)
    img = ndimage.imread(data_path + image, mode="L")
    img = misc.imresize(img, (img_rows, img_cols))
    data = np.asarray(img, dtype="int32")
    #np.reshape(data, (256, 256, 1))
    print data.shape
    #if i % 100 ==0:
        #misc.imsave(str(i)+'orig.png', data)
    #img = PImage.open(data_path + image)
    loadedImages.append(data)

X_train = np.asarray(loadedImages)
print X_train.shape


#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train.shape

X_train = X_train.reshape(len(X_train), img_rows, img_cols, 1)
#X_test = X_test.reshape(len(X_test), 28, 28, 1)

CNN_G = Sequential([
    Dense(512*7*7, input_dim=100, activation=LeakyReLU()),
    BatchNormalization(),
    Reshape((7, 7, 512)),
    UpSampling2D(),
    Convolution2D(64, 3, 3, border_mode='same', activation=LeakyReLU()),
    BatchNormalization(),
    UpSampling2D(),
    Convolution2D(32, 3, 3, border_mode='same', activation=LeakyReLU()),
    BatchNormalization(),
    Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid')
])

CNN_D = Sequential([
    Convolution2D(256, 5, 5, subsample=(2,2), border_mode='same', 
                  input_shape=(28, 28, 1), activation=LeakyReLU()),
    Convolution2D(512, 5, 5, subsample=(2,2), border_mode='same', activation=LeakyReLU()),
    Flatten(),
    Dense(256, activation=LeakyReLU()),
    Dense(1, activation = 'sigmoid')
])

CNN_D.compile(Adam(1e-3), "binary_crossentropy", metrics=['accuracy'])

def noise(bs): return np.random.rand(bs,100)

sz = len(X_train)//200
print np.random.permutation(X_train)[:sz].shape, CNN_G.predict(noise(sz)).shape
x1 = np.concatenate([np.random.permutation(X_train)[:sz], CNN_G.predict(noise(sz))])
CNN_D.fit(x1, [0]*sz + [1]*sz, batch_size=128, nb_epoch=1, verbose=2)

CNN_m = Sequential([CNN_G, CNN_D])
CNN_m.compile(Adam(1e-4), "binary_crossentropy", metrics=['accuracy'])

K.set_value(CNN_D.optimizer.lr, 1e-3)
K.set_value(CNN_m.optimizer.lr, 1e-3)


def plot_images(x_train, generator, save2file=False, fake=True, samples=16, noise=None, step=0):
    name = "cats"
    filename = name + '.png'
    if fake:
        if noise is None:
            noise = np.random.normal(0, 0.02, size=[samples, 100])
        else:
            filename = name + "fake" + "_%d.png" % step
        images = generator.predict(noise)
        print images[0][8][8], images[1][8][8]
    else:
        filename = name + "REAL" + "_%d.png" % step
        i = np.random.randint(0, x_train.shape[0], samples)
        print i
        images = x_train[i, :, :, :]

    plt.figure(figsize=(10,10))
    print "helo", images.shape[0]
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        image = images[i, :, :, :]
        #image = np.reshape(image, [img_rows, img_cols, channel])
        #comment below line while using color images
        image = np.reshape(image, (img_rows, img_cols))
        #image = np.reshape(image, [img_rows, img_cols
        #print image.shape
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()

def train(x_train, generator, discriminator, adversarial, train_steps=2000, batch_size=250, save_interval=0):
    noise_input = None
    if save_interval>0:
        noise_input = np.random.normal(0, 0.2, size=[16, 100])
    for i in range(train_steps):
        images_train = x_train[np.random.randint(0,
            x_train.shape[0], size=batch_size), :, :, :]
        noise = np.random.normal(0, 0.02, size=[batch_size, 100])
        images_fake = generator.predict(noise)
        print images_train.shape, images_fake.shape
        x = np.concatenate((images_train, images_fake))
        y = np.ones([2*batch_size, 1])
        y[batch_size:, :] = 0
        d_loss = discriminator.train_on_batch(x, y)

        y = np.ones([batch_size, 1])
        noise = np.random.normal(0, 0.02, size=[batch_size, 100])
        a_loss = adversarial.train_on_batch(noise, y)
        print d_loss, a_loss
        log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
        print(log_mesg)
        if save_interval>0:
            if (i+1)%save_interval==0:
                plot_images(x_train, generator, save2file=True, samples=noise_input.shape[0],\
                    noise=noise_input, step=(i+1))
                plot_images(x_train, generator, save2file=True, fake=False, samples=noise_input.shape[0],\
                    noise=noise_input, step=(i+1))
                
train(X_train, CNN_G, CNN_D, CNN_m, 10000, 250, 1000)
                


#dl,gl = train(CNN_D, CNN_G, CNN_m, 2500)

