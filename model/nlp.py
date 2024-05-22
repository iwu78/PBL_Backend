class ImageCaptioningModel:
    def __init__(self, image_path, captions_path, batch_size=64, img_size=224):
        # Initialize attributes
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

    # Other methods...

    def build_model(self):
        # Define the model architecture
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

        # Compile the model
        self.model = Model(inputs=[input1, input2], outputs=output)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def train_model(self, epochs=50):
        # Train the model
        train_generator = CustomDataGenerator(df=self.train, X_col='image', y_col='caption', batch_size=self.batch_size,
                                              directory=self.image_path, tokenizer=self.tokenizer, vocab_size=self.vocab_size,
                                              max_length=self.max_length, features=self.features)

        validation_generator = CustomDataGenerator(df=self.test, X_col='image', y_col='caption', batch_size=self.batch_size,
                                                   directory=self.image_path, tokenizer=self.tokenizer, vocab_size=self.vocab_size,
                                                   max_length=self.max_length, features=self.features)

        self.history = self.model.fit(train_generator, epochs=epochs, validation_data=validation_generator,
                                      callbacks=[checkpoint, earlystopping, learning_rate_reduction])

        # Save the model after training
        self.model.save('image_captioning_model.h5')

        self._plot_history()
