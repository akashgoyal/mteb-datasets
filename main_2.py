from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import mteb 


#
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
# Select tasks
tasks = mteb.get_tasks(tasks=["SCIDOCS"])
evaluation = mteb.MTEB(tasks=tasks)
# Force dataset to come from local path
for task in evaluation.tasks:
    print(task.metadata.name)
    # task.dataset = load_from_disk(f"./mteb_data/{task.metadata.name}")
    task.dataset = load_from_disk(f"./mteb_data/scidocs")
# Run evaluation
evaluation.run(model, output_folder="./results")
