import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") > 0.95:
            print("\nReached 95% accuracy so cancelling training.")
            self.model.stop_training = True

callbacks = myCallback()

data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

print(f"training_images shape: {training_images.shape}")
print(f"training_labels shape: {training_labels.shape}")

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.summary()

history = model.fit(training_images, training_labels, epochs=50,
                    callbacks=[callbacks])

model.evaluate(test_images, test_labels)

plt.plot(history.history["loss"])
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.grid(True)
plt.show()
#
#
# def plot_graphs(history, string):
#   plt.plot(history.history[string])
#   #plt.plot(history.history['val_'+string])
#   plt.xlabel("Epochs")
#   plt.ylabel(string)
#   #plt.legend([string, 'val_'+string])
#   plt.show()
#
# # Plot the accuracy and results
# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")
#

classifications = model.predict(test_images)
print(np.argmax(classifications[0]))
print(test_labels[0])

