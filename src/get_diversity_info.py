import pickle
import numpy as np
from tqdm import tqdm

with open("data/authors_info.pkl", "rb") as f:
    authors_info = pickle.load(f)    


# model_names = [, "msmarco-MiniLM-L6-cos-v5", "scibert_scivocab_uncased", "doc2vec", "wmd", "ql", "ql-rm3", "bm25", "bm25-rm3"]
# model_names = ["scibert-e-sara", "scibert-e-sara-v2", "specter", "msmarco-bert-base-dot-v5", "msmarco-MiniLM-L6-cos-v5", "scibert_scivocab_uncased", "all-MiniLM-L6-v2", "doc2vec", "wmd", "bm25", "bm25-rm3", "ql", "ql-rm3"]
model_names = ["scibert-e-sara", "specter", "msmarco-bert-base-dot-v5"]
for model_name in tqdm(model_names, desc="Model"):
    runfile_path = f"/runs//run.{model_name}.top-rank.trec"

    try:
        with open(runfile_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        continue
        
    top_10_authors = {}
    top_20_authors = {}
    top_100_authors = {}
    for line in lines:
        qid, _, doc_id, rank, _, _ = line.strip().split()
        if qid not in top_10_authors:
            top_10_authors[qid] = []
            top_20_authors[qid] = []
            top_100_authors[qid] = []
            
        if int(rank) <= 10:
            top_10_authors[qid].append(doc_id)
        if int(rank) <= 20:
            top_20_authors[qid].append(doc_id)
        if int(rank) <= 100:
            top_100_authors[qid].append(doc_id)
            

    exp_years_stds_10 = []
    exp_years_stds_20 = []
    exp_years_stds_100 = []
    exp_years_means_10 = []
    exp_years_means_20 = []
    exp_years_means_100 = []

    citations_stds_10 = []
    citations_stds_20 = []
    citations_stds_100 = []
    citations_means_10 = []
    citations_means_20 = []
    citations_means_100 = []

    works_count_stds_10 = []
    works_count_stds_20 = []
    works_count_stds_100 = []
    works_count_means_10 = []
    works_count_means_20 = []
    works_count_means_100 = []

    unique_ins_means_10 = []
    unique_ins_means_20 = []
    unique_ins_means_100 = []

    for qid, authors in top_10_authors.items():
        exp_years = [authors_info[author]["experience_years"] for author in authors]
        citations = [authors_info[author]["citations"] for author in authors]
        works_count = [authors_info[author]["works_count"] for author in authors]
        ins = [authors_info[author]["institution"] for author in authors if authors_info[author]["institution"] is not None]
        ins = set(ins)
        
        exp_years_stds_10.append(np.std(exp_years))
        exp_years_means_10.append(np.mean(exp_years))
        
        citations_stds_10.append(np.std(citations))
        citations_means_10.append(np.mean(citations))
        
        works_count_stds_10.append(np.std(works_count))
        works_count_means_10.append(np.mean(works_count))
        
        unique_ins_means_10.append(len(ins))
        
    for qid, authors in top_20_authors.items():
        exp_years = [authors_info[author]["experience_years"] for author in authors]
        citations = [authors_info[author]["citations"] for author in authors]
        works_count = [authors_info[author]["works_count"] for author in authors]
        ins = [authors_info[author]["institution"] for author in authors if authors_info[author]["institution"] is not None]
        ins = set(ins)
        
        exp_years_stds_20.append(np.std(exp_years))
        exp_years_means_20.append(np.mean(exp_years))
        
        citations_stds_20.append(np.std(citations))
        citations_means_20.append(np.mean(citations))
        
        works_count_stds_20.append(np.std(works_count))
        works_count_means_20.append(np.mean(works_count))
        
        unique_ins_means_20.append(len(ins))
        
    for qid, authors in top_100_authors.items():
        exp_years = [authors_info[author]["experience_years"] for author in authors]
        citations = [authors_info[author]["citations"] for author in authors]
        works_count = [authors_info[author]["works_count"] for author in authors]
        ins = [authors_info[author]["institution"] for author in authors if authors_info[author]["institution"] is not None]
        ins = set(ins)
        
        exp_years_stds_100.append(np.std(exp_years))
        exp_years_means_100.append(np.mean(exp_years))
        
        citations_stds_100.append(np.std(citations))
        citations_means_100.append(np.mean(citations))
        
        works_count_stds_100.append(np.std(works_count))
        works_count_means_100.append(np.mean(works_count))
        
        unique_ins_means_100.append(len(ins))


    print("Top 10")
    print("Experience years std:", np.mean(exp_years_stds_10))
    print("Experience years mean:", np.mean(exp_years_means_10))
    print("Citations std:", np.mean(citations_stds_10))
    print("Citations mean:", np.mean(citations_means_10))
    print("Works count std:", np.mean(works_count_stds_10))
    print("Works count mean:", np.mean(works_count_means_10))
    print("Unique institutions mean:", np.mean(unique_ins_means_10))

    print("Top 20")
    print("Experience years std:", np.mean(exp_years_stds_20))
    print("Experience years mean:", np.mean(exp_years_means_20))
    print("Citations std:", np.mean(citations_stds_20))
    print("Citations mean:", np.mean(citations_means_20))
    print("Works count std:", np.mean(works_count_stds_20))
    print("Works count mean:", np.mean(works_count_means_20))
    print("Unique institutions mean:", np.mean(unique_ins_means_20))

    print("Top 100")
    print("Experience years std:", np.mean(exp_years_stds_100))
    print("Experience years mean:", np.mean(exp_years_means_100))
    print("Citations std:", np.mean(citations_stds_100))
    print("Citations mean:", np.mean(citations_means_100))
    print("Works count std:", np.mean(works_count_stds_100))
    print("Works count mean:", np.mean(works_count_means_100))
    print("Unique institutions mean:", np.mean(unique_ins_means_100))
    

    with open(f"unique_ins_10_{model_name}.pkl", "wb") as f:
        pickle.dump(unique_ins_means_10, f)

    with open(f"unique_ins_20_{model_name}.pkl", "wb") as f:
        pickle.dump(unique_ins_means_20, f)

    with open(f"unique_ins_100_{model_name}.pkl", "wb") as f:
        pickle.dump(unique_ins_means_100, f)

    print()
    print(f"{np.mean(citations_means_10):0.4f} {np.mean(citations_stds_10):0.4f} {np.mean(citations_means_20):0.4f} {np.mean(citations_stds_20):0.4f} {np.mean(citations_means_100):0.4f} {np.mean(citations_stds_100):0.4f} {np.mean(works_count_means_10):0.4f} {np.mean(works_count_stds_10):0.4f} {np.mean(works_count_means_20):0.4f} {np.mean(works_count_stds_20):0.4f} {np.mean(works_count_means_100):0.4f} {np.mean(works_count_stds_100):0.4f} {np.mean(unique_ins_means_10):0.4f} {np.mean(unique_ins_means_20):0.4f} {np.mean(unique_ins_means_100):0.4f}")
    print(runfile_path)
    print("--------------------------------------------------")
