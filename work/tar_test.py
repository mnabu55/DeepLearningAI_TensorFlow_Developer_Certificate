import zipfile
import tensorflow as tf
import pathlib
import tensorflow as tf
import tarfile

# Unzip training dataset
t = tarfile.open("test.tar.gz", "r")
t.extractall("./test")

p_file = pathlib.Path('test/test.txt')

print(p_file)
print(type(p_file))
