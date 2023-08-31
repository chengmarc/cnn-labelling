# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
try:
    # Import core libraries
    import numpy as np
    import tensorflow as tf
    from keras.utils import to_categorical

    # Import libraries for utilities
    import os, gzip, struct, subprocess
    from PIL import Image
    from colorama import init, Fore
    init()
    print(Fore.GREEN + "All libraries imported.")

except:
    print("Dependencies missing, please use pip to install all dependencies:")
    print("numpy, tensorflow, keras, os, gzip, struct, subprocess, PIL, colorama")
    input('Press any key to quit.')
    exit()

# %% Import functions for MNIST dataset
# Credit for import functions: https://stackoverflow.com/questions/39969045
def load_dataset(path_dataset):       
   with gzip.open(path_dataset,'rb') as f:
     magic, size = struct.unpack(">II", f.read(8))
     nrows, ncols = struct.unpack(">II", f.read(8))
     data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
     data = data.reshape((size, nrows, ncols))
     return data

def load_label(path_label):
  with gzip.open(path_label,'rb') as f:
    magic, size = struct.unpack('>II', f.read(8))
    label = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
    return label

# %% Load MNIST data into system memory
script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)
import_path = script_dir + r"\MNIST\raw"

X_train = load_dataset(import_path + '\\' + 'train-images-idx3-ubyte.gz') / 255.0
Y_train = load_label(import_path + '\\' + 'train-labels-idx1-ubyte.gz')
Y_train = to_categorical(Y_train)

X_test = load_dataset(import_path + '\\' + 't10k-images-idx3-ubyte.gz') / 255.0
Y_test = load_label(import_path + '\\' + 't10k-labels-idx1-ubyte.gz')
Y_test = to_categorical(Y_test)
print(Fore.WHITE + "Data imported.")

# %% Model training
with tf.device('/CPU:0'):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5)
    print(Fore.GREEN + "Model training finished.")

# %% Model prediction
boolean = 'Y'
print(Fore.WHITE + "Please write a number in MS Paint and save it.")

while boolean == 'Y' or boolean == 'y':
    
    image = Image.new('L', (28, 28), color=0)
    image_path = 'test_image.png'
    image.save(image_path)

    process = subprocess.Popen(['mspaint', image_path])
    process.wait()

    modified_image = Image.open(image_path).convert('L')
    modified_image = np.array(modified_image)
    modified_image = modified_image.reshape(-1, 28, 28, 1)
    
    with tf.device('/CPU:0'): predictions = model.predict(modified_image)
    index = np.argmax(predictions)
    value = predictions[0, index]

    print(Fore.WHITE + 'According to the prediction of the trained model,')
    print(Fore.WHITE + 'the given number is', index, 'with a', value*100, '% confidence.')
    
    print("")
    boolean = input(Fore.WHITE + "Do you want to write another number? (Y/N)")
