from pathlib import Path
from pickle import load

import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


def generate_sequence(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')

        predict_x = model.predict(encoded)
        yhat = np.argmax(predict_x, axis=1)

        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break

        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)


def predict(text, sequence_length, model_path: Path):
    model = load_model(model_path / 'model.h5')
    tokenizer = load(open(model_path / 'tokenizer.pkl', 'rb'))

    generated = generate_sequence(model, tokenizer, sequence_length, text, 50)
    return {'text': text, 'generated': generated}
