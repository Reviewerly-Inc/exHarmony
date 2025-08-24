# exHarmony: Authorship and Citations for Benchmarking the Reviewer Assignment Problem

This repository contains the code and data for the paper "Authorship and Citations for Benchmarking the Reviewer Assignment Problem" and the "exHarmony" dataset.
It should be noted that due to the size of the dataset, we are unable to provide the full dataset in this repository.
Hence, the repository contains the codes for the sake of reproducibility and the data are available on:

* **Hugging Face Datasets**: [https://huggingface.co/datasets/Reviewerly/exHarmony](https://huggingface.co/datasets/Reviewerly/exHarmony)
* **Google Drive**: Follow the provided instructions in [Download from Google Drive](#option-2-download-from-google-drive) section.


## Dataset

### Option 1: Download from Hugging Face

You can directly load the dataset using the 🤗 `datasets` library:

```bash
pip install datasets
```

```python
from datasets import load_dataset

dataset = load_dataset("Reviewerly/exHarmony")
```

### Option 2: Download from Google Drive

To download the dataset from Google Drive, you can use the following commands:

Note: You need to install the `gdown` package to download the dataset.

```bash
pip install gdown
```

If you already have the `gdown` package installed, you can use the following commands to download the dataset:

```bash
cd exHarmony/
gdown --folder https://drive.google.com/drive/folders/1ZukiOXKn-Hdxdhd4oOMy57RVdyz-XTar?usp=sharing
```

Now you can find the dataset files in the `data` directory.

### Dataset Files

| Description            | File Name                          | File Size | Num Records | Format                                                            |
|------------------------|------------------------------------|-----------|-------------|-------------------------------------------------------------------|
| Collection             | papers_collection.jsonl            | 1.6 GB    | 1,204,150   | paper_id, title, abstract                                         |
| Test                   | papers_test.jsonl                  | 15 MB     | 9,771       | paper_id, title, abstract                                         |
| Test (judgable)        | papers_test_judgable.jsonl         | 14 MB     | 7,944       | paper_id, title, abstract                                         |
| Authors' Works Mapping | authors_works_collection_ids.jsonl | 222 MB    | 1,589,723   | author_id, list_of_authors_papers                                 |
| Authors' Information   | authors_info.jsonl                 | 225 MB    | 1,589,723   | author_id, citation, works_count, experience_years, institution   |

Each file is stored in JSON Lines format (`.jsonl`), where each line is a JSON object corresponding to a record.

### File Descriptions

It is now possible to access the exHarmony dataset in the `data` folder. Here are the files it contains:

- `papers_collection.jsonl`: Contains information about the papers that have been used to build the index for the author retrieval task.
- `papers_test.tsv`: Contains the test set for the author retrieval task.
- `papers_test_judgable.tsv`: Since the authors in the collection are those who have had any publications before time $\tau$, some authors might not exist in the collection. As such, they cannot be used as the gold standard for evaluation. Therefore, we remove any papers for which none of their authors exist in the test set and created this set.
    - Here is a sample row from the `papers_test.tsv` file which illustrates the structure in these three files:
      ```json
      {"id": "https://openalex.org/W4323317762", "title": "Sharding-Based Proof-of-Stake Blockchain Protocols: Key Components &amp; Probabilistic Security Analysis", "abstract": "Blockchain technology has been gaining great interest from a variety of sectors including healthcare, supply chain, and cryptocurrencies. However, Blockchain suffers from a limited ability to scale (i.e., low throughput and high latency). Several solutions have been proposed to tackle this. In particular, sharding has proved to be one of the most promising solutions to Blockchain's scalability issue. Sharding can be divided into two major categories: (1) Sharding-based Proof-of-Work (PoW) Blockchain protocols, and (2) Sharding-based Proof-of-Stake (PoS) Blockchain protocols. The two categories achieve good performances (i.e., good throughput with a reasonable latency), but raise security issues. This article focuses on the second category. In this paper, we start by introducing the key components of sharding-based PoS Blockchain protocols. We then briefly introduce two consensus mechanisms, namely PoS and practical Byzantine Fault Tolerance (pBFT), and discuss their use and limitations in the context of sharding-based Blockchain protocols. Next, we provide a probabilistic model to analyze the security of these protocols. More specifically, we compute the probability of committing a faulty block and measure the security by computing the number of years to fail. We achieve a number of years to fail of approximately 4000 in a network of 4000 nodes, 10 shards, and a shard resiliency of 33%."}
      ```

- `authors_works_collection_ids.jsonl`: Contains the mapping between authors and their works in the index.
    - Here is a sample row from the file:
      ```json
      {"id": "https://openalex.org/A5083262615", "works": ["https://openalex.org/W4323317762", "https://openalex.org/W4285189682", "https://openalex.org/W2994826096", "https://openalex.org/W3090464427", "https://openalex.org/W3039746697", "https://openalex.org/W2955700129", "https://openalex.org/W4313679422", "https://openalex.org/W4310681652", "https://openalex.org/W4311152538"]}
      ```
- `authors_info.jsonl`: Contains the information about the authors in the index.
    - Here is a sample item from the dictionary:
      ```json
      {"id": "https://openalex.org/A5083262615", "citations": 238, "works_count": 14, "experience_years": 5, "institution": "Université de Montréal"}
      ```

## Qrel Files

In this dataset, we approached the reviewer assignment problem by defining and evaluating multiple qrels, each
reflecting a different strategy for identifying relevant authors to assign to papers. These qrels were designed to
capture distinct perspectives on relevance, enabling a comprehensive evaluation of our retrieval models. We evaluated
five primary qrel sets, each representing unique relationships between papers and authors. The first qrel set considers
each paper's authors as relevant authors, while the second defines relevance based on authors of the most similar papers
within the test set. The third set uses citation information, marking as relevant the authors of works cited by each
paper. To refine the citation-based approach, the fourth set limits the relevant authors to those associated with the
top 10 most similar cited papers. Additionally, an aggregated set combines the first and second sets, broadening the
definition of relevance by combining authors of both the paper itself and the most similar paper. Each primary qrel was
further filtered to retain only “experts” — defined as authors with more than 15 publications — resulting in a total of
ten qrel sets. This multi-faceted approach allowed us to evaluate model performance across various interpretations of
relevance.

| Qrel Set       | Description                                                                | Number of Pairs | Filename                  |
|----------------|----------------------------------------------------------------------------|-----------------|---------------------------|
| Authors        | Authors of each paper are considered relevant.                             | 33,120          | `qrels.test.authors.tsv` |
| Cite           | Authors of the most similar paper in the test set are considered relevant. | 695,271         | `qrels.test.cite.tsv`    |
| SimCite        | Narrows citation-based relevance to the top-10 most similar cited papers.  | 202,904         | `qrels.test.simcite.tsv` |

Each of these sets is further filtered to include only “expert” authors, resulting in ten final qrel sets for a more
targeted evaluation of the retrieval models.

## Run Files
While you can easily reproduce the results by running the codes for each model, we also provide the run files for each one to make it easier for you to reevaluate the models. The run files contain the top 100 authors for each paper in the test set. All run files that have been used in the paper had been uploaded to Google Drive. You can download them by following the instructions below.

The models that have been used in the paper are as follows:

- `BM25`
- `Doc2Vec`
- `Word Movers' Distance`
- `BERT`: A BERT model that has been fine-tuned on the MS MARCO dataset.
- `MiniLM`: A MiniLM model that has been fine-tuned on the MS MARCO dataset.
- `SciBERT`
- `SPECTER`
- `SciBERT`: A SciBERT model that has been fine-tuned on scholarly literature.

### Download the run files

To download the run files from Google Drive, you can use the following commands:

```bash
cd exHarmony/
gdown --folder https://drive.google.com/drive/folders/1mOqE8EMV-WQNfCLtzO6oqvckNto3jzEs?usp=sharing
```

Now you have the run files in the `run_files` directory.

## Abstract

The peer review process is crucial for ensuring the quality and reliability of scholarly work, yet assigning suitable reviewers remains a significant challenge. Traditional manual methods are labor-intensive and often ineffective, leading to unconstructive or biased reviews. This paper introduces the exHarmony benchmark, designed to address these challenges by reimagining the Reviewer Assignment Problem (RAP) as a retrieval task. Utilizing the extensive data from OpenAlex, we propose a novel approach that considers an author as the best potential reviewer for their own paper. This allows us to evaluate and improve reviewer assignment without needing explicit labels. We benchmark various methods, including traditional lexical matching, static neural embeddings, and contextualized neural embeddings, and introduce evaluation metrics that assess both relevance and diversity. Our results indicate that while traditional methods perform reasonably well, contextualized embeddings trained on scholarly literature show the best performance. The findings underscore the importance of further research to enhance the diversity and effectiveness of reviewer assignments.

## Citation

If you use this resource, please cite our paper:

```
@inproceedings{ebrahimi2025exharmony,
  author = {Ebrahimi, Sajad and Salamat, Sara and Arabzadeh, Negar and Bashari, Mahdi and Bagheri, Ebrahim},
  title = {exHarmony: Authorship and Citations for Benchmarking the Reviewer Assignment Problem},
  year = {2025},
  isbn = {978-3-031-88713-0},
  publisher = {Springer-Verlag},
  doi = {10.1007/978-3-031-88714-7_1},
  booktitle = {Advances in Information Retrieval: 47th European Conference on Information Retrieval, ECIR 2025, Lucca, Italy, April 6–10, 2025, Proceedings, Part III},
  pages = {1–16},
}
```
