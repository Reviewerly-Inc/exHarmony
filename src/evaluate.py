import pytrec_eval
import sys
import argparse
import pickle

# Function to load TREC formatted file into a dictionary
def load_trec_file(file_path, is_qrel=True):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if is_qrel:
                qid, _, docid, rel = parts
                if qid not in data:
                    data[qid] = {}
                data[qid][docid] = int(rel)
            else:
                qid, _, docid, _, score, _ = parts
                if qid not in data:
                    data[qid] = {}
                data[qid][docid] = float(score)
    return data

parser = argparse.ArgumentParser(description='Calculate Information Retrieval Metrics')
parser.add_argument('--run', type=str, help='TREC run file path')
parser.add_argument('--qrels', type=str, help='TREC qrels file path')
parser.add_argument('--k', type=int, default=10, help='cut-off value')
args = parser.parse_args()

qrels = load_trec_file(args.qrels, is_qrel=True)
run = load_trec_file(args.run, is_qrel=False)

# create run_10, run_20, run_100 for evaluating rr
run_10 = {}
run_20 = {}
run_100 = {}
for qid, docs in run.items():
    run_10[qid] = {doc: score for doc, score in sorted(docs.items(), key=lambda x: x[1], reverse=True)[:10]}
    run_20[qid] = {doc: score for doc, score in sorted(docs.items(), key=lambda x: x[1], reverse=True)[:20]}
    run_100[qid] = {doc: score for doc, score in sorted(docs.items(), key=lambda x: x[1], reverse=True)[:100]}

# Initialize the evaluator with the required metrics
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {
    'ndcg_cut.10', 'ndcg_cut.20', 'ndcg_cut.100',
    'map_cut.10', 'map_cut.20', 'map_cut.100',
    'recall_10', 'recall_20', 'recall_100',
})
rr_evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank'})

# Evaluate the run
results = evaluator.evaluate(run)
results_rr_10 = rr_evaluator.evaluate(run_10)
results_rr_20 = rr_evaluator.evaluate(run_20)
results_rr_100 = rr_evaluator.evaluate(run_100)

# Aggregate the results
metrics = {
    'ndcg_cut_10': 0, 'ndcg_cut_20': 0, 'ndcg_cut_100': 0,
    'map_cut_10': 0, 'map_cut_20': 0, 'map_cut_100': 0,
    'recall_10': 0, 'recall_20': 0, 'recall_100': 0,
    'recip_rank_10': 0, 'recip_rank_20': 0, 'recip_rank_100': 0,
}
num_queries = len(results)

for query_id, query_measures in results.items():
    metrics['ndcg_cut_10'] += query_measures['ndcg_cut_10']
    metrics['ndcg_cut_20'] += query_measures['ndcg_cut_20']
    metrics['ndcg_cut_100'] += query_measures['ndcg_cut_100']
    metrics['map_cut_10'] += query_measures['map_cut_10']
    metrics['map_cut_20'] += query_measures['map_cut_20']
    metrics['map_cut_100'] += query_measures['map_cut_100']
    metrics['recall_10'] += query_measures['recall_10']
    metrics['recall_20'] += query_measures['recall_20']
    metrics['recall_100'] += query_measures['recall_100']
    
for query_id, query_measures in results_rr_10.items():
    metrics['recip_rank_10'] += query_measures['recip_rank']
    
for query_id, query_measures in results_rr_20.items():
    metrics['recip_rank_20'] += query_measures['recip_rank']
    
for query_id, query_measures in results_rr_100.items():
    metrics['recip_rank_100'] += query_measures['recip_rank']

# Average the metrics over all queries
for metric in metrics.keys():
    metrics[metric] /= num_queries

# Print the results
for metric, value in metrics.items():
    print(f'{metric}: {value:.4f}')
    
# for metric, value in metrics.items():
#     print(f'{value:.4f}', end=' ')
# print()
