import sys, os
sys.path.append('/home/04381/ymarathe/CS395T-DeepLearning-Fall17/Project1/model/tf_cnnvis/')
import numpy as np
np.random.seed(42)
import keras
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Input
from keras.models import Model
from keras.applications.xception import Xception
from keras.applications.vgg19 import VGG19
from keras import metrics
from keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, Flatten, Activation, Lambda, Input
import tensorflow as tf
import os
import keras.backend.tensorflow_backend as K

weights_dir = '/work/05148/picsou/maverick/project2/'
data_path = weights_dir
output_dir = '/work/05145/anikeshk/GAN_Expts/WassersteinGAN/visualizations_100_real_100_fake/'

K.set_learning_phase(0)
import tf_cnnvis

pretrained_model = VGG19(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
for layer in pretrained_model.layers:
    layer.trainable = False
    
x = pretrained_model.output
x = Conv2D(32, (1, 1), activation='relu')(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
predicted_years = Dense(12, activation='softmax')(x)

lr = 1e-4
def lr_schedule(epoch):
    return lr * (0.1 ** float(epoch / 10.0))

BATCH_SIZE = 1
IMG_SIZE = (64, 64)
real_data_path = data_path + 'keras_yearbook_less/'
gen_data_path = data_path + 'prog_fourbyfour/'
train_real = ImageDataGenerator().flow_from_directory(real_data_path + 'train/', target_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True)
train_gen  = ImageDataGenerator().flow_from_directory(gen_data_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True)
valid = ImageDataGenerator().flow_from_directory(real_data_path + 'valid/', target_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True)

def mixed_generator(ratio_real):
    while True:
        # Sample a batch from the real images
        # and one from the generated images
        real_x, real_y = train_real.next()
        if len(real_x) < 32:
            train_real.reset()
            real_x, real_y = train_real.next()
        gen_x, gen_y = train_gen.next()
        if len(gen_x) < 32:
            train_gen.reset()
            gen_x, gen_y = train_gen.next()
        # Sample ration_real percent of real images
        num_real = int(BATCH_SIZE * ratio_real)
        sampled_real_indices = np.random.choice(BATCH_SIZE, num_real, replace=False)
        sampled_real_x = real_x[sampled_real_indices]
        sampled_real_y = real_y[sampled_real_indices]
        # Sample 1-ratio_real percent of generated images
        num_gen = BATCH_SIZE - num_real
        sampled_gen_indices = np.random.choice(BATCH_SIZE, num_gen, replace=False)
        sampled_gen_x = gen_x[sampled_gen_indices]
        sampled_gen_y = gen_y[sampled_gen_indices]

        # Create mixed batch
        batch_x = np.concatenate((sampled_real_x, sampled_gen_x), axis=0)
        batch_y = np.concatenate((sampled_real_y, sampled_gen_y), axis=0)
        yield (batch_x, batch_y)

training = True
list_num_experiments = [15, 20]
layers = ['r', 'p', 'c']

for num_experiment in list_num_experiments:
    filename = "will2_{}.h5".format(num_experiment)
    model = Model(inputs=pretrained_model.input, outputs=predicted_years)
    model.compile(Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(weights_dir + "will2_{}.h5".format(num_experiment))
    batch_real, batch_real_y = next(mixed_generator(1.0))
    batch_fake, batch_fake_y  = next(mixed_generator(0.0))
    print(batch_real.shape)
    print(batch_fake.shape)
    input_img = model.input
    #with tf.device('/device:GPU:0'):
    #    K.set_session(tf.Session(config=tf.ConfigProto(allow_soft_placement = True)))
    print ("Starting experiment " + str(num_experiment))
    print("Using" + filename)
    print("IMAGE SIZE: ", IMG_SIZE)
    print("DATASET: ", gen_data_path)
    print("Output dir: ", output_dir)
    is_success = tf_cnnvis.activation_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {input_img : batch_real}, 
                                  layers=layers, path_logdir= output_dir + 'real/Log_Activation/Xception', path_outdir= output_dir+ 'real/Output_Activation/Xception')
    is_success = tf_cnnvis.activation_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {input_img : batch_fake}, 
                                  layers=layers, path_logdir= output_dir + 'fake/Log_Activation/Xception', path_outdir= output_dir+ 'fake/Output_Activation/Xception')

'''
is_success = tf_cnnvis.deconv_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {input_img : batch_x}, 
                                     layers=layers, path_logdir= output_dir + '/Log_Deconv/Xception', path_outdir= output_dir+ '/Output_DeConv/Xception')

is_success = tf_cnnvis.deepdream_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {input_img : batch_x}, 
  i                                   layers=layers, path_logdir= output_dir+ '/Log_DD/Xception', path_outdir= output_dir+ '/Output_DD/Xception')
'''
