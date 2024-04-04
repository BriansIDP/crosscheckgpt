import torch
import sys
import time
import json
import spacy
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI, SelfCheckLLMPrompt


target_LLM = sys.argv[1]
nlp = spacy.load("en_core_web_sm")
device = "xpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_dir = "/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/LLMknowledge/cache/"
llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
selfcheck = SelfCheckLLMPrompt(llm_model, device, cache_dir=cache_dir)
# selfcheck_nli = SelfCheckNLI(device=device, cache_dir=cache_dir)

selfcheck.model = selfcheck.model.eval().to(device)

# dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
with open("data/crosscheck/sentences_{}.json".format(target_LLM)) as fin:
    orig_dataset = json.load(fin)
    dataset = {}
    for wiki_bio_test_idx, passages in orig_dataset.items():
        dataset[wiki_bio_test_idx] = passages[:10]
        if len(dataset) >= 100:
            break

gpt3_dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
wiki_bio_test_idx = gpt3_dataset["evaluation"]["wiki_bio_test_idx"]
wiki_bio_text = gpt3_dataset["evaluation"]["wiki_bio_text"]
gpt3_text_samples = gpt3_dataset["evaluation"]["gpt3_text_samples"]
gpt3_datadict = {}
wikibio_ref = {}
for bioidx, text_samples in zip(*[wiki_bio_test_idx, gpt3_text_samples]):
    gpt3_datadict[str(bioidx)] = text_samples
for bioidx, text_samples in zip(*[wiki_bio_test_idx, wiki_bio_text]):
    wikibio_ref[str(bioidx)] = text_samples

diversity = {}

for wiki_bio_test_idx, passages in tqdm(dataset.items()):
    diversity[wiki_bio_test_idx] = []
    for passage in passages:
        sentences = [sent.text.strip() for sent in nlp(passage).sents]
        sent_scores_prompt = selfcheck.predict(
            sentences = sentences,
            sampled_passages = passages,
        )
        diversity[wiki_bio_test_idx].append(sum(sent_scores_prompt) / len(sent_scores_prompt))

with open("outputs/diversity_{}_selfcheck.json".format(target_LLM), "w") as fout:
    json.dump(diversity, fout, indent=4)
