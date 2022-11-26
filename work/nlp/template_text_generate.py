import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



DATA_FILE = "./sonnets.txt"

# open data file
with open(DATA_FILE, "r") as f:
    data = f.read()

corpus = data.lower().split("\n")


# parameter
max_word_len = 100
OOV_TOKEN = "<OOV>"

#tokenizer = Tokenizer(max_word_len, oov_token=OOV_TOKEN)
tokenizer = Tokenizer()

tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index) + 1

print(f"word index dictionary: {tokenizer.word_index}")
print(f"total words: {total_words}")
