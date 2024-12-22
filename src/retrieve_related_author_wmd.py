import os
from tqdm import tqdm
import time
import gensim.downloader as api
from gensim.utils import simple_preprocess
import pickle
import numpy as np
from collections import Counter
from annoy import AnnoyIndex

output_path = "output/WMD/"
run_output_path = os.path.join(output_path, 'Run_wmd.txt')

### Data files
data_folder = 'data'

# Load pre-trained word vectors
w2v_model = api.load('word2vec-google-news-300')
embedding_size = w2v_model.vector_size

# Step 1: Precompute Word Embeddings and Build Annoy Index
annoy_index = AnnoyIndex(embedding_size, 'euclidean')
word_to_index = {}
for i, word in enumerate(w2v_model.index_to_key):
    vector = w2v_model[word]
    annoy_index.add_item(i, vector)
    word_to_index[word] = i

annoy_index.build(10)  # Number of trees

# Load the corpus
print('Loading collection...')
pids = []
corpus = []
corpus_filepath = os.path.join(data_folder, 'papers_collection.pkl')
with open(corpus_filepath, 'rb') as f:
    works = pickle.load(f)
    for work in tqdm(works):
        pids.append(work['id'])
        corpus.append(f"{work['title']} {work['abstract']}")
        
    print(f'Number of documents: {len(corpus)}')


# Tokenize the corpus
tokenized_corpus_path = os.path.join(output_path, 'corpus_wmd.pkl')
if os.path.exists(tokenized_corpus_path):
    print('Loading tokenized corpus...')
    with open(tokenized_corpus_path, 'rb') as f:
        corpus_tokenized = pickle.load(f)
else:
    print('Tokenizing corpus...')
    corpus_tokenized = [simple_preprocess(doc) for doc in tqdm(corpus)]
    with open(tokenized_corpus_path, 'wb') as f:
        pickle.dump(corpus_tokenized, f, protocol=pickle.HIGHEST_PROTOCOL)

def preprocess(text):
    return [word for word in text.lower().split() if word in word_to_index]

def document_to_vector(document):
    word_counts = Counter(preprocess(document))
    doc_vector = np.zeros((len(word_counts), embedding_size))
    doc_weights = np.zeros(len(word_counts))
    
    for i, (word, count) in enumerate(word_counts.items()):
        doc_vector[i] = w2v_model[word]
        doc_weights[i] = count
        
    return doc_vector, doc_weights

def wmd_distance(doc1, doc2):
    doc1_vector, doc1_weights = document_to_vector(doc1)
    doc2_vector, doc2_weights = document_to_vector(doc2)
    
    # Normalize weights
    doc1_weights /= doc1_weights.sum()
    doc2_weights /= doc2_weights.sum()

    # Compute distance matrix
    distance_matrix = np.zeros((len(doc1_vector), len(doc2_vector)))
    for i, vec1 in enumerate(doc1_vector):
        for j, vec2 in enumerate(doc2_vector):
            distance_matrix[i, j] = np.linalg.norm(vec1 - vec2)
            
    # Solve the transportation problem
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    
    total_cost = distance_matrix[row_ind, col_ind].sum()
    
    return total_cost

# Initialize the Annoy index for the corpus
annoy_corpus_index = AnnoyIndex(embedding_size, 'euclidean')
corpus_vectors = []
for i, doc in enumerate(corpus_tokenized):
    word_vectors = [w2v_model[word] for word in doc if word in w2v_model.key_to_index]
    if word_vectors:
        doc_vector = np.mean(word_vectors, axis=0)
        annoy_corpus_index.add_item(i, doc_vector)
        corpus_vectors.append(doc_vector)
    else:
        # Handle empty word vectors case
        corpus_vectors.append(np.zeros(embedding_size))

annoy_corpus_index.build(10)  # Number of trees

# Read the queries
print('Loading queries...')
qids = []
queries = []
queries_filepath = os.path.join(data_folder, 'papers_test.pkl')
with open(queries_filepath, 'rb') as f:
    works = pickle.load(f)
    for work in tqdm(works):
        qids.append(work['id'])
        queries.append(work['title'])
        
    print(f'Number of queries: {len(queries)}')

# Tokenize the queries
queries_tokenized = [simple_preprocess(query) for query in queries]

results_path = os.path.join(output_path, 'results_wmd.pkl')

if os.path.exists(results_path):
    print('Loading results...')
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
else:
    # Perform WMD similarity search
    print('Performing WMD similarity search...')
    start_time = time.time()
    results = []
    for query in tqdm(queries_tokenized, desc='Searching'):
        word_vectors = [w2v_model[word] for word in query if word in w2v_model.key_to_index]
        if word_vectors:
            query_vector = np.mean(word_vectors, axis=0)
            nearest_neighbors = annoy_corpus_index.get_nns_by_vector(query_vector, 100, include_distances=True)
            results.append(list(zip(nearest_neighbors[0], nearest_neighbors[1])))
        else:
            results.append([])

        print(f'Search time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')

        # Save the results
        with open(results_path, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    

# Write the results to file
print('Writing the result to file...')
with open(run_output_path, 'w', encoding='utf-8') as fOut:
    for qid, result in enumerate(results):
        rank = 0
        for pid, score in result:
            fOut.write(f'{qids[qid]} Q0 {pids[pid]} {rank + 1} {score} Author_Retrieval\n')
            rank += 1