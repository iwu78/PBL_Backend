import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from textwrap import wrap
import matplotlib.pyplot as plt

# Load the tokenizer and max_length
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('max_length.txt', 'r') as f:
    max_length = int(f.read())

# Load the trained model
caption_model = load_model('model.h5')

# Load the DenseNet201 feature extractor model
fe = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, pooling='avg')

# Function to read and preprocess images
def readImage(path, img_size=224):
    img = load_img(path, color_mode='rgb', target_size=(img_size, img_size))
    img = img_to_array(img)
    img = img / 255.
    return img

# Function to get the image features
def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.
    img = np.expand_dims(img, axis=0)
    feature = fe.predict(img, verbose=0)
    return feature

# Function to map an index to a word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to predict the caption
def predict_caption(model, image_path, tokenizer, max_length):
    feature = extract_features(image_path)
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_pred = model.predict([feature, sequence])
        y_pred = np.argmax(y_pred)
        word = idx_to_word(y_pred, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

# Function to display images and their captions
def display_images(images, captions):
    plt.figure(figsize=(20, 20))
    n = 0
    for i in range(len(images)):
        n += 1
        plt.subplot(5, 5, n)
        plt.subplots_adjust(hspace=0.7, wspace=0.3)
        image = readImage(images[i])
        plt.imshow(image)
        plt.title("\n".join(wrap(captions[i], 20)))
        plt.axis("off")
    plt.show()

# List of new images to caption
image_paths = [
    'path_to_image1.jpg',
    'path_to_image2.jpg',
    # Add more image paths here
]

# Generate captions for the images
captions = []
for image_path in image_paths:
    caption = predict_caption(caption_model, image_path, tokenizer, max_length)
    captions.append(caption)

# Display the images with their captions
display_images(image_paths, captions)
