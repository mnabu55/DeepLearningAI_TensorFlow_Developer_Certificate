import tensorflow as tf
import numpy as np

# Generate the data. data shape: list ex) [1, 2, 3, ...]
dataset = tf.data.Dataset.range(10)

# Window. data shape: Window Dataset
dataset = dataset.window(5, shift=1, drop_remainder=True)

# Flatten the windows. data shape: ex) [1 2 3 ...]
dataset = dataset.flat_map(lambda window: window.batch(5))

# Create tuples with features and labels
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

# Shuffle the window
# buffer size should be more than the total number of windows
dataset = dataset.shuffle(buffer_size=10)

# Create batches of windows
dataset = dataset.batch(2).prefetch(1)

for x, y in dataset:
    print("x = ", x.numpy())
    print("y = ", y.numpy())
    print()
