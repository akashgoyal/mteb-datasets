import os
from datasets import load_dataset

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

# Download datasets
for k,v in ds_map.items():
    print(f"Downloading for Task:{k} Dataset:{v}")
    output_path = f"./mteb_datasets/{k}"
    if not os.path.exists(output_path):
        dataset = load_dataset(f"{v}", "default")
        dataset.save_to_disk(output_path)
    
