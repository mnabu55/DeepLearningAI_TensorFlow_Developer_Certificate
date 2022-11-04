import zipfile
import tensorflow as tf

# Unzip training dataset
local_zip = './test.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./')
zip_ref.close()
