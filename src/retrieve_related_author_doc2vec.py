import os
from tqdm import tqdm
import time
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.similarities import Similarity
from gensim.utils import simple_preprocess
import pickle


output_path = "output/Doc2Vec/"
run_output_path = os.path.join(output_path, 'Run_doc2vec.txt')

### Data files
data_folder = 'data'

### Read the corpus files, that contain all the passages. Store them in the corpus list
print('Loading collection...')
# corpus = {}
pids = []
corpus = []
corpus_filepath = os.path.join(data_folder, 'papers_collection.pkl')
with open(corpus_filepath, 'rb') as f:
    works = pickle.load(f)
    for work in tqdm(works):
        pids.append(work['id'])
        corpus.append(f"{work['title']} {work['abstract']}")
        
    print(f'Number of documents: {len(corpus)}')    
    

print('Loading author mapping...')
author_mapping = {}
author_mapping_filepath = os.path.join(data_folder, 'authors_works_collection_ids.pkl')
with open(author_mapping_filepath, 'rb') as f:
    author_work_ids = pickle.load(f)
    
    for awi in author_work_ids:
        author_id = awi['id']
        works = awi['works']
        for work in works:
            if work not in author_mapping:
                author_mapping[work] = []
            author_mapping[work].append(author_id)


tagged_corpus_path = os.path.join(output_path, 'corpus_doc2vec.pkl')
if os.path.exists(tagged_corpus_path):
    print('Loading tokenized corpus...')
    with open(tagged_corpus_path, 'rb') as f:
        tagged_corpus = pickle.load(f)
else:
    print('Tokenizing corpus...')
    tagged_corpus = [TaggedDocument(simple_preprocess(doc), [i]) for i, doc in tqdm(enumerate(corpus), total=len(corpus))]

    # save the tagged corpus
    with open(os.path.join(output_path, 'corpus_doc2vec.pkl'), 'wb') as f:
        pickle.dump(tagged_corpus, f, protocol=pickle.HIGHEST_PROTOCOL)


trained_model_path = os.path.join(output_path, "doc2vec.model")
if os.path.exists(trained_model_path):
    print('Loading Doc2Vec model...')
    d2v_model = Doc2Vec.load(trained_model_path)
else:
    # Train a Doc2Vec model
    print('Training Doc2Vec model...')
    d2v_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=16)
    d2v_model.build_vocab(tagged_corpus)
    d2v_model.train(tagged_corpus, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
    d2v_model.save(trained_model_path)

# Create document vectors for the corpus
corpus_vectors_path = os.path.join(output_path, 'corpus_vectors.pkl')
if os.path.exists(corpus_vectors_path):
    print('Loading document vectors...')
    with open(corpus_vectors_path, 'rb') as f:
        corpus_vectors = pickle.load(f)
else:
    print('Creating document vectors for the corpus...')
    corpus_vectors = [d2v_model.infer_vector(doc.words) for doc in tqdm(tagged_corpus)]
    
    with open(corpus_vectors_path, 'wb') as f:
        pickle.dump(corpus_vectors, f, protocol=pickle.HIGHEST_PROTOCOL)

# Create a similarity index
print('Creating similarity index...')
index = Similarity(output_path, corpus_vectors, num_features=100)

# Read the queries
print('Loading queries...')
qids = []
queries = []
queries_filepath = os.path.join(data_folder, 'papers_test.pkl')
with open(queries_filepath, 'rb') as f:
    works = pickle.load(f)
    for work in tqdm(works):
        qids.append(work['id'])
        queries.append(f"{work['title']} {work['abstract']}")
        
    print(f'Number of queries: {len(queries)}')

# Tokenize the queries
queries_tokenized = [simple_preprocess(query) for query in queries]

results_path = os.path.join(output_path, 'results_doc2vec.pkl')
if os.path.exists(results_path):
    print('Loading results...')
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
else:
    # Perform Doc2Vec similarity search
    print('Performing Doc2Vec similarity search...')
    start_time = time.time()
    results = []
    for query in queries_tokenized:
        query_vector = d2v_model.infer_vector(query)
        sims = index[query_vector]
        top_k_indices = sims.argsort()[-1000:][::-1]
        results.append([(idx, sims[idx]) for idx in top_k_indices])
    print(f'Search time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')

    # save the results
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
