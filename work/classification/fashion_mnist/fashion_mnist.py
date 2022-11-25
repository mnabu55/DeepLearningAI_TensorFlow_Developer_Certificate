import tensorflow as tf

print("tensorflow: ", tf.__version__)

fmist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

# Create a model with DNN
model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile the model
model_1.compile(optimizer='Adam',
                loss='sparse_categorical_crossentropy',
                metrics=["accuracy"])

# model_1.summary()

model_1.fit(training_images, training_labels, epochs=5)

model_1.evaluate(test_images, test_labels)


# Create a model with Conv2D
model_2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile the model
model_2.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model_2.summary()

# train the model
#model_2.fit(tf.expand_dims(training_images, axis=-1), training_labels, epochs=5)
model_2.fit(training_images, training_labels, epochs=5)

# evaluate the model
#model_2.evaluate(tf.expand_dims(test_images, axis=-1), test_labels)
model_2.evaluate(test_images, test_labels)
