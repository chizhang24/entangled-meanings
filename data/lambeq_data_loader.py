import spacy
import numpy as np
import os
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

nlp = spacy.load("en_core_web_md")
nltk.download('punkt')

# Get the relative path of the lambeq dataset
lambeq_path = os.path.join(os.path.dirname(__file__), "lambeq.txt")

def read_data(filename):
  labels, sentences = [], []
  with open(filename) as f:
    for line in f:
      t = int(line[0])
      if t == 1:
        labels.append(t)
      else:
        labels.append(-1)
      sentences.append(line[1:].strip())
  return labels, sentences

def spacy_load():
  labels, doc = read_data(lambeq_path)
  sentence_vectors = [nlp(s).vector for s in doc]
  X = np.array(sentence_vectors)
  Y = np.array(labels)
  return X, Y


def average_vector(sentence, model):
  vectors = [model.wv[word] for word in sentence if word in model.wv]
  if vectors:
      return np.mean(vectors, axis=0)
  else:
      return np.zeros(model.vector_size)
  

def wv_load():
  labels, sentences = read_data(lambeq_path)
  tokenized_sentences = [word_tokenize(s.lower()) for s in sentences]
  model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
  sentence_vectors = [average_vector(sentence, model) for sentence in tokenized_sentences]
  X = np.array(sentence_vectors)
  Y = np.array(labels)
  return X, Y


