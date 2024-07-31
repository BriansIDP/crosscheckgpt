import torch
import sys
import time
import json
import spacy
from datasets import load_dataset
from tqdm import tqdm

from crosscheckgpt.modeling_crosscheck import CrossCheckLLMPrompt
from crosscheckgpt.modeling_crosscheck import CrossCheckQuestionsLLMPrompt
from crosscheckgpt.prompts import *


target_LLM = sys.argv[1]

template_dict = {
    "mistral": mistral,
    "llama2": llama2,
    "vicuna13b": vicuna,
    "llama213b": llama2,
    "vicuna": vicuna,
    "llama2lm": llama2lm,
    "llama213blm": llama2lm,
    "starling": starling,
    "falcon": falcon,
    "beluga": beluga,
    "openorca": openorca,
    "mistral1": mistral,
    "zephyr": zephyr,
}
model_paths = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral1": "mistralai/Mistral-7B-Instruct-v0.1",
    "llama2": "/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/LLMweakTOstrong/ckpt/llama-2-7b-chat-hf",
    "llama213b": "/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/LLMweakTOstrong/ckpt/llama-2-13b-chat-hf",
    "vicuna": "/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/LLMweakTOstrong/ckpt/vicuna-7b-v1.5",
    "vicuna13b": "/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/LLMweakTOstrong/ckpt/vicuna-13b-v1.5",
    "llama2lm": "/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/LLMweakTOstrong/ckpt/llama-2-7b-hf",
    "llama213blm": "/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/LLMweakTOstrong/ckpt/llama-2-13b-hf",
    "starling": "berkeley-nest/Starling-LM-7B-alpha",
    "falcon": "tiiuae/falcon-7b-instruct",
    "bloom": "bigscience/bloom-7b1",
    "beluga": "stabilityai/StableBeluga-7B",
    "openorca": "Open-Orca/Mistral-7B-OpenOrca",
    "zephyr": "HuggingFaceH4/zephyr-7b-beta",
}

nlp = spacy.load("en_core_web_sm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
cache_dir = "/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/LLMknowledge/cache/"
implicit = False

if not implicit:
    selfcheck = CrossCheckLLMPrompt(llm_model, device, cache_dir=cache_dir)
    # selfcheck_nli = SelfCheckNLI(device=device, cache_dir=cache_dir)
# else:
#     selfcheck = CrossCheckQuestionsLLMPrompt(llm_model, device, cache_dir=cache_dir, cot=False)

# dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
with open("data/crosscheck/onebest_{}.json".format(target_LLM)) as fin:
    dataset = json.load(fin)

gpt3_dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", cache_dir="./data/")
wiki_bio_test_idx = gpt3_dataset["evaluation"]["wiki_bio_test_idx"]
wiki_bio_text = gpt3_dataset["evaluation"]["wiki_bio_text"]
gpt3_text_samples = gpt3_dataset["evaluation"]["gpt3_text_samples"]
gpt3_datadict = {}
wikibio_ref = {}
for bioidx, text_samples in zip(*[wiki_bio_test_idx, gpt3_text_samples]):
    gpt3_datadict[str(bioidx)] = text_samples
for bioidx, text_samples in zip(*[wiki_bio_test_idx, wiki_bio_text]):
    wikibio_ref[str(bioidx)] = text_samples

crosscheckLLMs = ["mistral", "llama2", "vicuna", "zephyr", "beluga", "starling", "openorca", "llama2lm"]
# crosscheckLLMs = []

if target_LLM not in crosscheckLLMs and target_LLM != "gpt3":
    crosscheckLLMs.append(target_LLM)
# crosscheckLLMs = [target_LLM]
# crosscheckLLMs = ["ref"]

with open("data/wikipedia-biography-dataset/wikipedia-biography-dataset/test/test.title") as fin:
    titles = [title.strip() for title in fin.readlines()]

crosscheck_sents = {}
for LLMtype in crosscheckLLMs:
    if LLMtype != "gpt3" and LLMtype != "ref":
        with open("data/crosscheck/sentences_{}.json".format(LLMtype)) as fin:
            crosscheck_sents[LLMtype] = json.load(fin)
    else:
        crosscheck_sents[LLMtype] = gpt3_datadict

crossLLMs = crosscheck_sents.keys()

if implicit:
    results = {}
    with torch.no_grad():
        for LLMtype in crossLLMs:
            llm_model = model_paths[LLMtype]
            template = template_dict[LLMtype]
            selfcheck = CrossCheckQuestionsLLMPrompt(llm_model, device, cache_dir=cache_dir, cot=False, prompt_template=template)
            print("start checking {}".format(LLMtype))
            for wiki_bio_test_idx, passage in tqdm(dataset.items()):
                sentences = [sent.text.strip() for sent in nlp(passage[0]).sents]
                sent_scores_prompt = selfcheck.predict(
                    sentences = sentences,
                    person=titles[int(wiki_bio_test_idx)]
                )
                if wiki_bio_test_idx not in results:
                    results[wiki_bio_test_idx] = {}
                results[wiki_bio_test_idx][LLMtype] = sent_scores_prompt
            selfcheck.model.to(torch.device("cpu"))
            del selfcheck
    results = list(results.values())
else:        
    results = []
    with torch.no_grad():
        for wiki_bio_test_idx, passage in tqdm(dataset.items()):
            # Split into sentences
            sentences = [sent.text.strip() for sent in nlp(passage[0]).sents]
            new_result = {}
            # Get cross check results
            for LLMtype in crossLLMs:
                if LLMtype == "gpt3":
                    LLM_text_samples = gpt3_datadict[str(wiki_bio_test_idx)]
                elif LLMtype == "ref":
                    LLM_text_samples = [wikibio_ref[str(wiki_bio_test_idx)]]
                else:
                    LLM_text_samples = [sent for sent in crosscheck_sents[LLMtype][str(wiki_bio_test_idx)]]
                sent_scores_prompt = selfcheck.predict(
                    sentences = sentences,
                    sampled_passages = LLM_text_samples,
                )
                import pdb; pdb.set_trace()
                new_result[LLMtype] = sent_scores_prompt
            
            results.append(new_result)

with open("outputs/crosscheck_prompt_{}{}.json".format(target_LLM, "_implicit" if implicit else ""), "w") as fout:
    json.dump(results, fout, indent=4)
