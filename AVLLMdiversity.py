import torch
import sys
import time
import json
import spacy
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI, SelfCheckLLMPrompt


target_path = sys.argv[1]  # Path to the AVLLM outputs
target_LLM = sys.argv[2]  # name of the target AVLLM
nlp = spacy.load("en_core_web_sm")
device = "xpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_dir = "/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/LLMknowledge/cache/"
llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
selfcheck = SelfCheckLLMPrompt(llm_model, device, cache_dir=cache_dir)
# selfcheck_nli = SelfCheckNLI(device=device, cache_dir=cache_dir)

selfcheck.model = selfcheck.model.eval().to(device)

# dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
with open(target_path) as fin:
    dataset = json.load(fin)

diversity = {}

for video_idx, passages in tqdm(dataset.items()):
    diversity[video_idx] = []
    for passage in passages:
        sentences = [sent.text.strip() for sent in nlp(passage).sents]
        sent_scores_prompt = selfcheck.predict(
            sentences = sentences,
            sampled_passages = passages,
        )
        diversity[video_idx].append(sum(sent_scores_prompt) / len(sent_scores_prompt))

with open("outputs/diversity_{}_selfcheck.json".format(target_LLM), "w") as fout:
    json.dump(diversity, fout, indent=4)
