import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence, to_categorical, plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Reshape, Embedding, LSTM, Dropout, add, concatenate
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle
from textwrap import wrap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

plt.rcParams['font.size'] = 12
sns.set_style("dark")
warnings.filterwarnings('ignore')


class ImageCaptioningModel:
    def __init__(self, image_path, captions_path, batch_size=64, img_size=224):
        self.image_path = image_path
        self.captions_path = captions_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.data = None
        self.tokenizer = None
        self.vocab_size = None
        self.max_length = None
        self.features = {}
        self.model = None
        self.history = None

    def load_data(self):
        self.data = pd.read_csv(self.captions_path)
        self.data = self._text_preprocessing(self.data)
        captions = self.data['caption'].tolist()

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(captions)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.max_length = max(len(caption.split()) for caption in captions)

        images = self.data['image'].unique().tolist()
        nimages = len(images)
        split_index = round(0.85 * nimages)
        train_images = images[:split_index]
        val_images = images[split_index:]

        self.train = self.data[self.data['image'].isin(train_images)].reset_index(drop=True)
        self.test = self.data[self.data['image'].isin(val_images)].reset_index(drop=True)

    def _text_preprocessing(self, data):
        data['caption'] = data['caption'].apply(lambda x: x.lower())
        data['caption'] = data['caption'].apply(lambda x: x.replace("[^A-Za-z]", ""))
        data['caption'] = data['caption'].apply(lambda x: x.replace("\s+", " "))
        data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))
        data['caption'] = "startseq " + data['caption'] + " endseq"
        return data

    def extract_features(self):
        model = DenseNet201(weights='imagenet', include_top=False, pooling='avg')
        for image in tqdm(self.data['image'].unique().tolist()):
            img = load_img(os.path.join(self.image_path, image), target_size=(self.img_size, self.img_size))
            img = img_to_array(img)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            feature = model.predict(img, verbose=0)
            self.features[image] = feature

    def build_model(self):
        input1 = Input(shape=(1920,))
        input2 = Input(shape=(self.max_length,))

        img_features = Dense(256, activation='relu')(input1)
        img_features_reshaped = Reshape((1, 256), input_shape=(256,))(img_features)

        sentence_features = Embedding(self.vocab_size, 256, mask_zero=False)(input2)
        merged = concatenate([img_features_reshaped, sentence_features], axis=1)
        sentence_features = LSTM(256)(merged)
        x = Dropout(0.5)(sentence_features)
        x = add([x, img_features])
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.vocab_size, activation='softmax')(x)

        self.model = Model(inputs=[input1, input2], outputs=output)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def train_model(self, epochs=50):
        train_generator = CustomDataGenerator(df=self.train, X_col='image', y_col='caption', batch_size=self.batch_size,
                                              directory=self.image_path, tokenizer=self.tokenizer, vocab_size=self.vocab_size,
                                              max_length=self.max_length, features=self.features)

        validation_generator = CustomDataGenerator(df=self.test, X_col='image', y_col='caption', batch_size=self.batch_size,
                                                   directory=self.image_path, tokenizer=self.tokenizer, vocab_size=self.vocab_size,
                                                   max_length=self.max_length, features=self.features)

        model_name = "model.h5"
        checkpoint = ModelCheckpoint(model_name, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, restore_best_weights=True)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.2, min_lr=0.00000001)

        self.history = self.model.fit(train_generator, epochs=epochs, validation_data=validation_generator,
                                      callbacks=[checkpoint, earlystopping, learning_rate_reduction])

        self._plot_history()

        # Save the tokenizer and max_length
        with open('tokenizer.pkl', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('max_length.txt', 'w') as f:
            f.write(str(self.max_length))

    def _plot_history(self):
        plt.figure(figsize=(20, 8))
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

    def predict_caption(self, image_path):
        feature = self._extract_feature_for_prediction(image_path)
        in_text = "startseq"
        for _ in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            y_pred = self.model.predict([feature, sequence])
            y_pred = np.argmax(y_pred)
            word = self._idx_to_word(y_pred)
            if word is None:
                break
            in_text += " " + word
            if word == 'endseq':
                break
        return in_text

    def _extract_feature_for_prediction(self, image_path):
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        model = DenseNet201(weights='imagenet', include_top=False, pooling='avg')
        feature = model.predict(img, verbose=0)
        return feature

    def _idx_to_word(self, integer):
        for word, index in self.tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def display_images(self, image_paths, captions):
        plt.figure(figsize=(20, 20))
        n = 0
        for i in range(len(image_paths)):
            n += 1
            plt.subplot(5, 5, n)
            plt.subplots_adjust(hspace=0.7, wspace=0.3)
            image = self._read_image(image_paths[i])
            plt.imshow(image)
            plt.title("\n".join(wrap(captions[i], 20)))
            plt.axis("off")
        plt.show()

    def _read_image(self, path):
        img = load_img(path, color_mode='rgb', target_size=(self.img_size, self.img_size))
        img = img_to_array(img)
        img = img / 255.0
        return img


class CustomDataGenerator(Sequence):
    def __init__(self, df, X_col, y_col, batch_size, directory, tokenizer, vocab_size, max_length, features, shuffle=True):
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.directory = directory
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.features = features
        self.shuffle = shuffle
        self.n = len(self.df)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return self.n // self.batch_size

    def __getitem__(self, index):
        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size, :]
        X1, X2, y = self.__get_data(batch)
        return (X1, X2), y

    def __get_data(self, batch):
        X1, X2, y = [], [], []
        images = batch[self.X_col].tolist()
        for image in images:
            feature = self.features[image][0]
            captions = batch.loc[batch[self.X_col] == image, self.y_col].tolist()
            for caption in captions:
                seq = self.tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
        X1, X2, y = np.array(X1), np.array(X2), np.array(y)
        return X1, X2, y


def init_nlp():
    image_captioning = ImageCaptioningModel(image_path='../input/flickr8k/Images', captions_path="../input/flickr8k/captions.txt")
    image_captioning.load_data()
    image_captioning.extract_features()
    image_captioning.build_model()
    image_captioning.train_model(epochs=50)
    return image_captioning

# Example usage:
# image_captioning = init_nlp()
# caption = image_captioning.predict_caption('path_to_image.jpg')
# print(caption)
