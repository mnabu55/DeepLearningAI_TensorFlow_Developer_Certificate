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