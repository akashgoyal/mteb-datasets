from datasets import load_dataset

# Download SciDocs
dataset = load_dataset("mteb/scidocs", "default")
dataset.save_to_disk("./mteb_data/scidocs")
