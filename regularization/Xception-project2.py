
# coding: utf-8

# In[1]:

import numpy as np
rand_seed = 42
print rand_seed
np.random.seed(rand_seed)
from keras import regularizers
import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Input
from keras.models import Model
#from keras.applications.xception import Xception
#from keras.applications.vgg19 import VGG19
from keras import metrics
from keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, Flatten, Activation, Lambda, Input
#from .imagenet_utils import _obtain_input_shape
import tensorflow as tf
np.random.seed(rand_seed)
from keras.utils.data_utils import get_file
from PIL import Image
import PIL.Image

# In[2]:
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
BATCH_SIZE = 32
IMG_SIZE = (64, 64)
real_data_path = '/work/05148/picsou/maverick/project2/keras_yearbook_less/'
gen_data_path = '/work/05148/picsou/maverick/project2/prog_fourbyfour/'   #'/work/05148/picsou/maverick/project2/prog_fourbyfour/'
train_real = ImageDataGenerator().flow_from_directory(real_data_path+'train/', target_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True)
train_gen  = ImageDataGenerator().flow_from_directory(gen_data_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True)
valid = ImageDataGenerator().flow_from_directory(real_data_path+'valid/', target_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True)

def VGG19(reg, include_top=False, weights=None,
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=12):

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape


    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(reg), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(reg), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(reg), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(reg), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), kernel_regularizer=regularizers.l2(reg), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), kernel_regularizer=regularizers.l2(reg), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), kernel_regularizer=regularizers.l2(reg), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), kernel_regularizer=regularizers.l2(reg),activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), kernel_regularizer=regularizers.l2(reg), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), kernel_regularizer=regularizers.l2(reg),activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), kernel_regularizer=regularizers.l2(reg),activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), kernel_regularizer=regularizers.l2(reg),activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), kernel_regularizer=regularizers.l2(reg),activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), kernel_regularizer=regularizers.l2(reg),activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), kernel_regularizer=regularizers.l2(reg),activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), kernel_regularizer=regularizers.l2(reg),activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg19')

    # load weights: Ignore uptil line 160 
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    file_hash='cbe5617147190e668d6c5d5026f83318')
        else:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    elif weights is not None:
        model.load_weights(weights)
    

    return model

'''    
x = pretrained_model.output
x = Conv2D(32, (1, 1), activation='relu')(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
predicted_years = Dense(12, activation='softmax')(x)
'''

lr = 1e-4
def lr_schedule(epoch):
    return lr * (0.1 ** float(epoch / 10.0))

# In[3]:

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


# In[6]:

'''import matplotlib.pyplot as plt
m = mixed_generator(xxx)
for i in range(32):
    x,y = next(m)
    plt.imshow(x[i]/255)
    plt.show()'''


# In[ ]:

training = True
num_experiment = 21
ratio = 1.0
for rf in [0, 0.00001, 0.000001]:#[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    filename = "anik_fake4by4image_fakevalid_{}.h5".format(rf)
    pretrained_model = VGG19(rf, include_top=False, weights='imagenet', input_shape=(64, 64, 3))
    x = pretrained_model.output
    x = Conv2D(32, (1, 1), kernel_regularizer=regularizers.l2(rf), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(64, kernel_regularizer=regularizers.l2(rf), activation='relu')(x)
    predicted_years = Dense(12, activation='softmax')(x)
    model = Model(inputs=pretrained_model.input, outputs=predicted_years)
    model.compile(Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    #model.load_weights("will2_{}.h5".format(3))
    
    #model = Model(inputs=pretrained_model.input, outputs=predicted_years)
    
    #model = VGG19(rf, include_top=False, weights=None, input_shape=(64, 64, 3))
    #model.compile(Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    mixed_gen = mixed_generator(ratio)
    
    with tf.device('/gpu:0'):
        steps_per_epoch = 100
        print ("Starting experiment " + str(num_experiment))
        print("IMAGE SIZE: ", IMG_SIZE)
        print("DATASET: ", gen_data_path)
        print("RATIO: ", ratio)
        model.fit_generator(mixed_gen,
                            steps_per_epoch = steps_per_epoch, 
                            epochs = 10,                                
                            validation_data = valid, 
                            validation_steps = steps_per_epoch / 5,
                            callbacks=[LearningRateScheduler(lr_schedule)]
                           )
        print("Saved " + filename)
    num_experiment += 1
    del model


# # Experiment 3:
# Train the baseline model on 20 epochs on real training dataset
# Epoch 20/20
# 2000/2000 [==============================] - 206s - loss: 1.2179 - acc: 0.5625 - val_loss: 1.6750 - val_acc: 0.4242
# 
# 
# # Experiment 4:
# 
# Finetune the baseline model with weights of #3 on 20 epochs on real training dataset
# 
# Epoch 20/20:
# 1000/1000 [==============================] - 103s - loss: 1.0578 - acc: 0.6214 - val_loss: 1.6515 - val_acc: 0.4364
# 
# Val accuracy stays the same as in experiement 3 (expected behavior)
# 
# # Experiment 5:
# 
# Finetune the baseline model with weights of #3 on 20 epochs on generated 2x2 dataset
# 
# Epoch 20/20: 1000/1000 [==============================] - 102s - loss: 0.7678 - acc: 0.7217 - val_loss: 2.8005 - val_acc: 0.3150
# 
# Val accuracy is constant at around 30-32%
# 
# Train accuracy increases from 44% to 72% -> Overfitting of the generated dataset
# 
# # Experiment 6:
# 
# Finetune the baseline model with weights of #3 on 20 epochs on generated 4x4 dataset
# Epoch 20/20: 1000/1000 [==============================] - 102s - loss: 0.8157 - acc: 0.6947 - val_loss: 2.1215 - val_acc: 0.3682
# 
# # Experiment 7:
# 
# Finetune the baseline model with weights of #3 on 20 epochs on mixed of 20% of real data and the rest of generated 4x4 dataset
# Epoch 17/20: 1000/1000 [==============================] - 196s - loss: 0.9903 - acc: 0.6357 - val_loss: 1.7202 - val_acc: 0.4145
# 
# # Experiment 8:
# 
# Finetune the baseline model with weights of #3 on 20 epochs on mixed of 40% of real data and the rest of generated 4x4 dataset
# Epoch 20/20: 1000/1000 [==============================] - 187s - loss: 1.0766 - acc: 0.6041 - val_loss: 1.6567 - val_acc: 0.4249
# 
# # Experiment 9:
# 
# Finetune the baseline model with weights of #3 on 20 epochs on mixed of 60% of real data and the rest of generated 4x4 dataset
# Epoch 19/20: 1000/1000 [==============================] - 194s - loss: 1.1441 - acc: 0.5810 - val_loss: 1.5977 - val_acc: 0.4404
# 
# 
# # Experiment 10:
# 
# Finetune the baseline model with weights of #3 on 20 epochs on mixed of 80% of real data and the rest of generated 4x4 dataset
# Epoch 19/20: 1000/1000 [==============================] - 193s - loss: 1.1494 - acc: 0.5826 - val_loss: 1.6176 - val_acc: 0.4455
# 
# # Experiment 11:
# 
# Train the baseline model on 20 epochs on mixed of 20% of real data and the rest of generated 4x4 dataset
# Epoch 20/20: 1000/1000 [==============================] - 193s - loss: 1.1843 - acc: 0.5882 - val_loss: 2.0275 - val_acc: 0.3698
# 
# # Experiment 12:
# 
# Train the baseline model on 20 epochs on mixed of 40% of real data and the rest of generated 4x4 dataset
# Epoch 19/20
# 1000/1000 [==============================] - 199s - loss: 1.3602 - acc: 0.5124 - val_loss: 1.8474 - val_acc: 0.3546
# 
# 
# # Experiment 13:
# 
# Train the baseline model on 20 epochs on mixed of 60% of real data and the rest of generated 4x4 dataset
# Epoch 20/20: 1000/1000 [==============================] - 193s - loss: 1.5509 - acc: 0.4422 - val_loss: 1.8595 - val_acc: 0.3452
# 
# # Experiment 14:
# 
# Train the baseline model on 20 epochs on mixed of 80% of real data and the rest of generated 4x4 dataset
# 
# 
# # Experiment 15-20
# Finetune the network #3 with 0, 20, 40, 60, 80, 100% of real images on 4x4 dataset
# 
# # Experiment 21 - 26
# 
# Finetune the network #3 with 0, 20, 40, 60, 80, 100% of real images on "global" dataset
# 
# 
# 
# 

# In[ ]:



