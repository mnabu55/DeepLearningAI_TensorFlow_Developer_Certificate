import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# load imdb reviews dataset
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

# get the train and test datasets
train_data, test_data = imdb['train'], imdb['test']

# Parameters
vocab_size = 10000
oov_tok = "<OOV>"
max_length = 120
embedded_dim = 16
trunc_type = 'post'

sentences = []

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize the tokenizer class
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

# generate the word index for the training sentences
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# generate and pad the training sequences
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)


# build a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedded_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=["accuracy"])

# model summary
model.summary()

# train the model
num_epochs = 10
model.fit(padded, labels, epochs=num_epochs,
          validation_data=(testing_padded, testing_labels_final))
