from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import  requests
import json

response = requests.get("https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json")
open("sarcasm.json", "wb").write(response.content)

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

# create Tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")

# Generate the word index dictionary
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
print(f"number of words in word_index: {len(word_index)}\n")

# generate padded sequences
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding="post")

# print sample headlie
index = 2
print(f"sample headline: {sentences[index]}")
print(f"sequence: {sequences[index]}")
print(f"padded sequence: {padded[index]}")


