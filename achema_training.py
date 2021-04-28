import os


import tensorflow as tf


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.applications.vgg16 import VGG16


"""
import keras as ke
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model

from keras.applications.vgg16 import VGG16
"""

train_dir = 'E:/TUKL_MSc_ComputerScience/Hackathons/Achema/KEEN Hackathon/KEEN Hackathon/02_Challenge by TU Dortmund/Training'
valid_dir = 'E:/TUKL_MSc_ComputerScience/Hackathons/Achema/KEEN Hackathon/KEEN Hackathon/02_Challenge by TU Dortmund/Validation'

#print(len(os.listdir(train_dir+'\Fluten')))


# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 20, class_mode = 'binary', target_size = (224, 224))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory( valid_dir,  batch_size = 20, class_mode = 'binary', target_size = (224, 224))


# Loading the Base Model VGG16
base_model = VGG16(input_shape = (224, 224, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')

# Since we don't have to train all the layers, make them non_traninable
for layer in base_model.layers:
    layer.trainable = False

# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])

#steps per epoch = number of batches = 10315/20
vgghist = model.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 515, epochs = 2,verbose=1)
