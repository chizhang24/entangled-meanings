import spacy
import numpy as np
import nltk
from nltk.tokenize import word_tokenize


nlp = spacy.load("en_core_web_md")
nltk.download('punkt')

# Change the path to the provided Amazon data
amazon_data_path = "/Users/charles/Documents/dcsp/QClassifier/cin/dataset/small_amazon_reviews.txt"

def read_data(filename):
    labels, sentences = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # Remove newline character and any leading/trailing whitespace
            label = int(line[-1])  # The label is the last character
            if label == 1:
                labels.append(label) 
            else:
                labels.append(-1)
            sentence = line[:-2].strip()  # Exclude the last 2 characters (label and tab)
            sentences.append(sentence)
    return labels, sentences

def spacy_load():
    labels, doc = read_data(amazon_data_path)
    sentence_vectors = [nlp(s).vector for s in doc]
    X = np.array(sentence_vectors)
    Y = np.array(labels)
    return X, Y

class Word2VecModel:
    def __init__(self, sentences, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.build_vocab(sentences)
        self.train(sentences)

    def build_vocab(self, sentences):
        self.vocab = {}
        for sentence in sentences:
            for word in sentence:
                if word in self.vocab:
                    self.vocab[word] += 1
                else:
                    self.vocab[word] = 1
        self.vocab = {word: count for word, count in self.vocab.items() if count >= self.min_count}

    def train(self, sentences):
        self.word_vectors = {word: np.random.uniform(-0.5, 0.5, self.vector_size) for word in self.vocab}
        for sentence in sentences:
            for word in sentence:
                if word in self.vocab:
                    # Here we are simulating training
                    self.word_vectors[word] += np.random.uniform(-0.5, 0.5, self.vector_size)

    def wv(self, word):
        return self.word_vectors.get(word, np.zeros(self.vector_size))

def average_vector(sentence, model):
    vectors = [model.wv(word) for word in sentence]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def wv_load():
    labels, sentences = read_data(amazon_data_path)
    tokenized_sentences = [word_tokenize(s.lower()) for s in sentences]
    model = Word2VecModel(sentences=tokenized_sentences)
    sentence_vectors = [average_vector(sentence, model) for sentence in tokenized_sentences]
    X = np.array(sentence_vectors)
    Y = np.array(labels)
    return X, Y

# Example usage:
# if __name__ == "__main__":
#     X_spacy, Y_spacy = spacy_load()
#     X_wv, Y_wv = wv_load()

#     print("SpaCy Vectors:")
#     print(X_spacy)
#     print("Labels:")
#     print(Y_spacy)

#     print("Word2Vec Vectors:")
#     print(X_wv)
#     print("Labels:")
#     print(Y_wv)