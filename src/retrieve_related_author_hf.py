import faiss
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm, trange
import time
import torch
import math
import pickle

model_name = "sentence-transformers/msmarco-bert-base-dot-v5"
output_path = f"output/{model_name.split('/')[-1]}_pretrained/"
run_output_path = os.path.join(output_path, 'Run.txt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(output_path, exist_ok=True)

# We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
model = SentenceTransformer(model_name, device=device)
embedding_dim = model.get_sentence_embedding_dimension()
print(f'{model_name} model loaded.')

### Data files
data_folder = 'data'

### Read the corpus files, that contain all the passages. Store them in the corpus dict
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


n = math.ceil(len(corpus) / 1000000)

# check whether faiss index exists
if os.path.exists(os.path.join(output_path, 'faiss.index')):
    index = faiss.read_index(os.path.join(output_path, 'faiss.index'))
    print(f'Index loaded. Index size: {index.ntotal}')
else:
    print('Index does not exist. Creating a new index...')
    for x in trange(n, desc='Encoding corpus'):
        corpus_embeddings = model.encode(corpus[x * 1000000:(x + 1) * 1000000], convert_to_tensor=True, show_progress_bar=True, batch_size=256)
        torch.save(corpus_embeddings, os.path.join(output_path, f'corpus_tensor_{x}.pt'))

    index = faiss.IndexFlatL2(embedding_dim)
    print(index.is_trained)

    for i in trange(n, desc='Creating index'):
        all_corpus = torch.load(os.path.join(output_path, f'corpus_tensor_{i}.pt'), map_location=torch.device(device)).detach().cpu().numpy()
        index.add(all_corpus)

    print(index.ntotal)
    faiss.write_index(index, os.path.join(output_path, 'faiss.index'))
    print(f'Index saved. Index size: {index.ntotal}')

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

top_k = 1000
xq = model.encode(queries)
start_time = time.time()
D, I = index.search(xq, top_k)
print(f'Search time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')

# save D and I
print('Saving the results...')
with open(os.path.join(output_path, 'results.pkl'), 'wb') as f:
    pickle.dump((D, I), f, protocol=pickle.HIGHEST_PROTOCOL)
    
# loading D and I
# with open(os.path.join(output_path, 'results.pkl'), 'rb') as f:
#     D, I = pickle.load(f)

print('Writing the result to file...')
with open(run_output_path, 'w', encoding='utf-8') as fOut:
    for qid in range(len(I)):
        author_count = {}
        for rank in range(top_k):
            score = D[qid][rank]
            
            retrieved_work = pids[I[qid][rank]]
            if retrieved_work not in author_mapping:
                continue
            for author in author_mapping[retrieved_work]:
                if author not in author_count:
                    author_count[author] = 0
                fOut.write(f'{qids[qid]} Q0 {author}_{author_count[author]} {rank + 1} {score} Author_Retrieval\n')
                author_count[author] += 1
                