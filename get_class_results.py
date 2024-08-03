import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap.umap_ as umap
from Data import lambeq_data_loader, amazon_data_loader
# Import the necessary functions from your classifier file
from quantum_classifier import main as qc_main










def load_data(dataset:str, vectorizer:str):
    if dataset =='lambeq':
        if vectorizer == 'wv':
            X, Y = lambeq_data_loader.wv_load()
            return X, Y
        if vectorizer == 'spacy':
            X, Y = lambeq_data_loader.spacy_load()
            return X, Y
    if dataset == 'amazon':
        if vectorizer == 'wv':
            X, Y = amazon_data_loader.wv_load()
            return X, Y
        if vectorizer == 'spacy':
            X, Y = amazon_data_loader.spacy_load()
            return X, Y
        


def reduce_dimensions(data, labels, method, method_name:str, dataset_name: str, vectorizer:str):
    print(f"Reducing dimensions... Dataset: {dataset_name}, Vectorizer: {vectorizer}, Reduction: {method_name}. Input shape: {data.shape}")
    if method_name == 'LDA':
        n_classes = 2
        n_features = data.shape[1]
        n_components = min(n_features, n_classes - 1)
        method = LDA(n_components=n_components)
        result = method.fit_transform(data, labels)
    elif method_name == 'PCA':
        method.fit(data)
        explained_variance = np.cumsum(method.explained_variance_ratio_)
        n_components = np.argmax(explained_variance >= 0.95) + 1  # Retain 95% variance
        method = PCA(n_components=n_components)
        result = method.fit_transform(data)
    elif method_name == 'NONE':
        result = data
    else:
        result = method.fit_transform(data)
    
    # if result.shape[1] == 0:
    #     print(f"Warning: {method_name} reduced data to 0 dimensions. Using original data.")
    #     return data#
    
    print(f"Reduced dimensions... Dataset: {dataset_name}, Vectorizer: {vectorizer}, Reduction: {method_name}. Output shape: {result.shape}")
    return result#, result.shape[1]


dataset_list = ['lambeq', 'amazon']
reduction_list = ['PCA', 'TSNE', 'LDA', 'UMAP', 'NONE'] 
encoding_list = ['amplitude', 'dc']
vectorizer_list = ['wv', 'spacy']




reduction_dict = {
    'PCA': PCA(),
    'TSNE': TSNE(),
    'UMAP': umap.UMAP(),
    'LDA': LDA(),
    'NONE': None,
}


reps = 10
steps = 100

results = []

# Iterate through all combinations
for dataset in dataset_list:
    for vectorizer in vectorizer_list:
        for reduction in reduction_list:
            for encoding in encoding_list:
                try:
                    print(f"\nProcessing Dataset: {dataset} Vectorizer:{vectorizer} Reduction: {reduction}")
                    # Reduce dimensions
                    X, Y = load_data(dataset, vectorizer)
                    reduced_data= reduce_dimensions(X, Y, reduction_dict[reduction], reduction, dataset, vectorizer)
                    
                    # Apply classification to reduced data
                    print(f"\n Classifying Dataset: {dataset} Vectorizer:{vectorizer} Reduction: {reduction} Quantum Encoding: {encoding}")
                    classifier_results = qc_main(X=reduced_data, Y=Y, encoding=encoding, reps=reps, steps=steps)
                    
                    # Store results
                    results.append({
                        'dataset': dataset,
                        'vectorizer': vectorizer,
                        'reduction': reduction,
                        'original dimensions': X.shape,
                        'reduced_dimensions': reduced_data.shape,
                        'quantum encoding': encoding,
                        'num_qubits': classifier_results[0],
                        'accuracy_train': classifier_results[1],
                        'accuracy_val': classifier_results[2],
                        'accuracy_test': classifier_results[3],
                        'bias': classifier_results[4],
                    })
                except Exception as e:
                    print(f"Error processing {dataset} with {reduction}: {str(e)}")





results_df = pd.DataFrame(results)

# Save or display the results
results_df.to_csv('classification_results.csv')
print(results_df.head())

