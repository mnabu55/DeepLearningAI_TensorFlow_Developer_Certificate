import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import json
import numpy as np
import zipfile

response = requests.get("https://nlp.stanford.edu/data/glove.twitter.27B.zip")
open("glove.zip", "wb").write(response.content)

local_zip = "glove.zip"
zip_ref = zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("./")
zip_ref.close()
