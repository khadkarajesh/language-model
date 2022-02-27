import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')


def load(filename):
    with open(filename, "r") as f:
        return f.read()


def clean(document):
    stop_words = stopwords.words('english')
    porter = PorterStemmer()
    words = word_tokenize(document)
    words = [token.lower() for token in words if token.isalpha()]
    words = [token for token in words if token not in stop_words]
    words = [porter.stem(token) for token in words]
    return words


def save(lines, filename):
    data = '\n'.join(lines)
    with open(filename, "w") as f:
        f.write(data)


def generate_sequences(words, save_to):
    length = 50 + 1
    sequences = list()
    for i in range(length, len(words)):
        seq = words[i - length:i]
        line = ' '.join(seq)
        sequences.append(line)
    save(sequences, save_to)


def preprocess(input_document, output_document):
    doc = load(input_document)
    tokens = clean(doc)
    generate_sequences(tokens, output_document)
