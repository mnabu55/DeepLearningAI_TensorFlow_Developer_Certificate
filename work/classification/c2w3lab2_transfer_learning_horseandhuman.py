import tensorflow as tf
import pathlib

model_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
#data_dir = tf.keras.utils.get_file('/tmp/inceptionv3.h5', origin=model_url, untar=False)
#data_dir = tf.keras.utils.get_file(origin=model_url, untar=False)

import requests
response = requests.get(model_url)
open("tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5", "wb").write(response.content)


from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers

# Set the weights file you downloaded into a variable
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Initialize the base model.
# Set the input shape and remove the dense layers.
pre_trained_model = InceptionV3(input_shape = (150, 150, 3),
                                include_top = False,
                                weights = None)

# Load the pre-trained weights you downloaded.
pre_trained_model.load_weights(local_weights_file)

# Freeze the weights of the layers.
for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()


# Callbacks
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > 0.999:
            print("\nReach")
            self.model.stop_training = True


# Choose `mixed_7` as the last layer of your base model
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

# Append the dense network to the base model
model = Model(inputs=pre_trained_model.input, outputs=x)

# Print the model summary. See your dense network connected at the end.
model.summary()

# Set the training parameters
model.compile(optimizer = RMSprop(learning_rate=0.0001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Extract the archive
zip_ref = zipfile.ZipFile("./horse-or-human.zip", 'r')
zip_ref.extractall("tmp/training")

zip_ref = zipfile.ZipFile("./validation-horse-or-human.zip", 'r')
zip_ref.extractall("tmp/validation")

zip_ref.close()

# Define our example directories and files
base_dir = 'tmp/'

train_dir = os.path.join(base_dir, 'training')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with training cat pictures
train_horses_dir = os.path.join(train_dir, 'horses')

# Directory with training dog pictures
train_humans_dir = os.path.join(train_dir, 'humans')

# Directory with validation cat pictures
validation_horses_dir = os.path.join(validation_dir, 'horses')

# Directory with validation dog pictures
validation_humans_dir = os.path.join(validation_dir, 'humans')

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.)

# Note that the validation data should not be augmented!
validation_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                    batch_size=32,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  validation_datagen.flow_from_directory(directory=validation_dir,
                                                          batch_size=32,
                                                          class_mode='binary',
                                                          target_size=(150, 150))

callbacks = myCallback()

# Train the model.
history = model.fit(
    train_generator,
    validation_data = validation_generator,
    epochs = 100,
    verbose = 2,
    callbacks=callbacks
)


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()
