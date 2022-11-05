import zipfile
import tensorflow as tf

# Unzip training dataset
local_zip = './zip.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./')
zip_ref.close()
