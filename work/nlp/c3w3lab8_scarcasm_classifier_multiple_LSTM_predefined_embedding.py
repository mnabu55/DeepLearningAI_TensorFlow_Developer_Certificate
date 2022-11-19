import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import json
import numpy as np
import zipfile

# response = requests.get("https://nlp.stanford.edu/data/glove.twitter.27B.zip")
# open("glove.zip", "wb").write(response.content)

# local_zip = "glove.zip"
# zip_ref = zipfile.ZipFile(local_zip, "r")
# zip_ref.extractall("./")
# zip_ref.close()

# response = requests.get("https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json")
# open("sarcasm.json", "wb").write(response.content)

with open("./sarcasm.json", "r") as f:
    datastore = json.load(f)

print(datastore[0])
print(datastore[20000])

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item["headline"])
    labels.append(item["is_sarcastic"])
    urls.append(item["article_link"])

print("len(sentences): ", len(sentences))

# Hyperparameters
training_size = 20000
vocab_size = 10000
max_length = 32
#embedding_dim = 16
embedding_dim = 25
trunc_type = "post"
padding_type = "post"
oov_token = "<OOV>"


# split training and test data
training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]

# define tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)

tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# generate padded sequences
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# training_padded = np.array(training_padded)
# testing_padded = np.array(testing_padded)
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)


# using pre-defined embeddings
GLOVE_FILE = "glove.twitter.27B.25d.txt"

GLOVE_EMBEDDINGS = {}

with open(GLOVE_FILE) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        GLOVE_EMBEDDINGS[word] = coefs
f.close()

test_word = 'dog'
test_vector = GLOVE_EMBEDDINGS[test_word]
print(f"Vector representation of word {test_word} looks like this:\n\n{test_vector}")
print(f"\nEach word vector has shape: {test_vector.shape}")

EMBEDDINGS_MATRIX = np.zeros((vocab_size + 1, embedding_dim))
for word, i in word_index.items():
    if i > vocab_size - 1:
        break
    else:
        embedding_vector = GLOVE_EMBEDDINGS.get(word)
        if embedding_vector is not None:
            EMBEDDINGS_MATRIX[i] = embedding_vector


# build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, weights=[EMBEDDINGS_MATRIX], trainable=False),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(24, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# print summary
model.summary()

# compile the model
adam = tf.keras.optimizers.Adam(learning_rate=0.000008,
                                beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=["accuracy"])

# train the model
num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels),
                    verbose=2)

import matplotlib.pyplot as plt


# Plot utility
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


# Plot the accuracy and loss
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")