# import tensorflow dataset
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# load imdb reviews dataset
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

print(info)

for example in imdb['train'].take(2):
    print(example)

# get the train and test datasets
train_data, test_data = imdb['train'], imdb['test']

# initialize sentences and labels lists
training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s, l in train_data:
    training_sentences.append(s.numpy().decode('utf8'))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(s.numpy().decode('utf8'))
    testing_labels.append(l.numpy())

# convert labels list to numpy array
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

print(f'len(training_labels_final): {len(training_labels_final)}')
print(f'len(testing_labels_final): {len(testing_labels_final)}')

# Parameters
vocab_size = 10000
max_length = 120
embedded_dim = 16
trunc_type = 'post'
oov_tok = "<OOV>"

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize the tokenizer class
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

# generate the word index for the training sentences
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# generate and pad the training sequences
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

# generate and pad the testing sequences
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, truncating=trunc_type)

print(padded[0])

# build a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedded_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=["accuracy"])

# model summary
model.summary()

print(f"padded shape: {padded.shape}")

# train the model
num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs,
          validation_data=(testing_padded, testing_labels_final))

# get the embedding layer from the model and weights
embedding_layer = model.layers[0]
embedding_weights = embedding_layer.get_weights()[0]

# get the index-word dictionary
reverse_word_index =  tokenizer.index_word

import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for word_num in range(1, vocab_size):
    word_name = reverse_word_index[word_num]
    word_embedding = embedding_weights[word_num]
    out_m.write(word_name + "\n")
    out_v.write('\t'.join([str(x) for x in word_embedding]) + "\n")

out_v.close()
out_m.close()


