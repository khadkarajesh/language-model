from pathlib import Path
from pickle import dump

from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from numpy import array
from tensorflow.keras.utils import to_categorical

MODEL_NAME = "model.h5"
TOKENIZER_NAME = "tokenizer.pkl"


def train(doc, epochs=1):
    lines = doc.split('\n')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)
    vocab_size = len(tokenizer.word_index) + 1

    sequences = array(sequences)
    X, y = sequences[:, :-1], sequences[:, -1]
    y = to_categorical(y, num_classes=vocab_size)
    seq_length = X.shape[1]

    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, batch_size=128, epochs=epochs)

    model.save(MODEL_NAME)
    dump(tokenizer, open(TOKENIZER_NAME, 'wb'))
    return {'model_path': str(Path.cwd() / MODEL_NAME), 'tokenizer_path': str(Path.cwd() / TOKENIZER_NAME)}
