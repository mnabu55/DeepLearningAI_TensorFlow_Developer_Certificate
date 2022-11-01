import tensorflow as tf
fmist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmist.load_data()

