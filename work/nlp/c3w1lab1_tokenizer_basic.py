from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!',

]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")

# Generate indices for each word in the corpus
tokenizer.fit_on_texts(sentences)

# Get the indices and print it
word_index = tokenizer.word_index
print("\nWord Index: ", word_index)

# Generate list of token sequences
sequences = tokenizer.texts_to_sequences(sentences)
print("\nSequences: ", sequences)

padded = pad_sequences(sequences, maxlen=10)
print("\nPadded Sequences: ", padded)