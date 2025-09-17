from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import mteb 

#
ds_map = {
    "ArguAna": "mteb/arguana",
    "CQADupstackRetrieval": "mteb/cqadupstack-retrieval",
    "ClimateFEVER": "mteb/climate-fever",
    "DBPedia": "mteb/dbpedia",
    "FEVER": "mteb/fever",
    # "FiQA2018": "mteb/FiQA2018-NL",
    "HotpotQA": "mteb/hotpotqa",
    "MSMARCO": "mteb/msmarco",
    "NFCorpus": "mteb/nfcorpus",
    "NQ": "mteb/nq",
    # "Quora-NL": "mteb/Quora-NL",
    "SCIDOCS": "mteb/scidocs",
    "SciFact": "mteb/scifact",
    "TRECCOVID": "mteb/trec-covid",
    "Touche2020": "mteb/touche2020"
}

#
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
# Select tasks
tasks = mteb.get_tasks(tasks=list(ds_map.keys()))
evaluation = mteb.MTEB(tasks=tasks)
# Force dataset to come from local path
for task in evaluation.tasks:
    print(task.metadata.name)
    task.dataset = load_from_disk(f"./mteb_datasets/{task.metadata.name}")
    # task.dataset = load_from_disk(f"./mteb_data/scidocs")
# Run evaluation
evaluation.run(model, output_folder="./results")
