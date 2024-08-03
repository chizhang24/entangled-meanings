# Entangled-Meanings
GitHub repo  for the IEEE QCE24 poster with the same title. 

## Classification Tasks

### Environment

```
pennylane 0.36.0
numpy 1.26.4 (PCA, tsne and LDA used)
scikit-learn 1.4.2
gensim 4.1.2 (Word2Vec used)
spacy 3.7.2
umap-learn 0.5.5 (UMAP used)
scipy 1.12.0 (must use version < 1.13.0, otherwise there will be conflicts with gensim)
```

### Datasets
The `lambeq` dataset is stored in `/Datasets/lambeq.txt` and the Amazon dataset is stored in `/Datasets/small_amazon_reviews.txt`. 

We load the lambeq dataset and vectorize the text using the python script `/Datasets/lambeq_data_loader`, while for the Amazon review dataset we use `/Datasets/amazon_data_loader`

### Quantum Encoding Algorithms

We implemented amplitude encoding and the divide-and-conquer encoding from [A divide-and-conquer algorithm for quantum state preparation](https://www.nature.com/articles/s41598-021-85474-1). The code for amplitude encoding is in `/QuantumEncodings/amp_enc.py`, and the code for divide-and-conquer encoding is in `/QuantumEncodings/dc_enc.py`. 


The code for the training process is in `/quantum_classifier.py`, by calling the `main()` function. 

### Dimension Reduction

We applied dimension reductions like `tsne`, `PCA`, `UMAP` and `LDA` in the python script `/get_class_results.py`. And by executing `get_class_results.py`, we can get the results in Table 1 in the poster.


### Results
The results in Table 1 in the poster is stored in `/classification_results.csv`.
 
 
## Ambiguity Resolution 

All the code and results for the ambiguity resolution task in the poster are in the jupyter notebook `/disambiguation.ipynb`. 

### Environment

```
qiskit 1.1.1
qiskit-aer 0.14.2
qiskit-machine-learning 0.7.2
numpy 1.26.4
```

