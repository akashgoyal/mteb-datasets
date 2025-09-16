# import torch
import mteb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# retrieval_tasks_eng_v1 = []
# tasks = mteb.get_tasks(task_types=["Retrieval"], languages=["eng"]) #, domains=["Legal"])
# print(tasks)
# eng_retrieval_benchmark = mteb.get_benchmark("MTEB(eng, v1)")
# print(eng_retrieval_benchmark)
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}_retrieval")
