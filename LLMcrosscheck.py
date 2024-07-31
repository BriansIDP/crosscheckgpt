import torch
import json
import spacy
from datasets import load_dataset
from tqdm import tqdm

from crosscheckgpt.modeling_crosscheck import CrossCheckLLMPrompt
from crosscheckgpt.modeling_crosscheck import CrossCheckQuestionsLLMPrompt
from crosscheckgpt.prompts import *

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

implicit = True

if not implicit:
    selfcheck = SelfCheckLLMPrompt(llm_model, device, cache_dir=cache_dir)
    # selfcheck = SelfCheckNLI(device=device, cache_dir=cache_dir, threeway=True)

dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")

with open("data/wikipedia-biography-dataset/wikipedia-biography-dataset/test/test.title") as fin:
    titles = [title.strip() for title in fin.readlines()]

crosscheckLLMs = ["mistral", "llama2", "vicuna", "zephyr", "beluga"]
# crosscheckLLMs = ["vicuna"]

crosscheck_sents = {}
for LLMtype in crosscheckLLMs:
    with open("data/crosscheck/sentences_{}.json".format(LLMtype)) as fin:
        crosscheck_sents[LLMtype] = json.load(fin)

crossLLMs = crosscheck_sents.keys()

results = []
labels = []
with torch.no_grad():
    if implicit:
        results = {}
        for LLMtype in crossLLMs:
            llm_model = model_paths[LLMtype]
            # template = template_dict[LLMtype]
            template = "{}"
            selfcheck = CrossCheckQuestionsLLMPrompt(llm_model, device, cache_dir=cache_dir, cot=False, prompt_template=template)
            print("start checking {}".format(LLMtype))
            for i in tqdm(range(dataset.num_rows["evaluation"])):
                gpt3_text = dataset["evaluation"][i]["gpt3_text"]
                wiki_bio_text = dataset["evaluation"][i]["wiki_bio_text"]
                gpt3_sentences = dataset["evaluation"][i]["gpt3_sentences"]
                annotation = dataset["evaluation"][i]["annotation"]
                wiki_bio_test_idx = dataset["evaluation"][i]["wiki_bio_test_idx"]
                gpt3_text_samples = dataset["evaluation"][i]["gpt3_text_samples"]
                sent_scores_prompt = selfcheck.predict(
                    sentences = gpt3_sentences,
                    person=titles[int(wiki_bio_test_idx)]
                )
                if wiki_bio_test_idx not in results:
                    results[wiki_bio_test_idx] = {}
                results[wiki_bio_test_idx][LLMtype] = sent_scores_prompt
            selfcheck.model.to(torch.device("cpu"))
            del selfcheck
        results = list(results.values())
    else:
        for i in tqdm(range(dataset.num_rows["evaluation"])):
            gpt3_text = dataset["evaluation"][i]["gpt3_text"]
            wiki_bio_text = dataset["evaluation"][i]["wiki_bio_text"]
            gpt3_sentences = dataset["evaluation"][i]["gpt3_sentences"]
            annotation = dataset["evaluation"][i]["annotation"]
            wiki_bio_test_idx = dataset["evaluation"][i]["wiki_bio_test_idx"]
            gpt3_text_samples = dataset["evaluation"][i]["gpt3_text_samples"]
        
            new_result = {}
            # Get GPT-3 result
            sent_scores_prompt = selfcheck.predict(
                sentences = gpt3_sentences,
                sampled_passages = gpt3_text_samples,
            )
            new_result["gpt3"] = sent_scores_prompt
            for LLMtype in crossLLMs:
                LLM_text_samples = [sent for sent in crosscheck_sents[LLMtype][str(wiki_bio_test_idx)]]
                print(len(LLM_text_samples))
                sent_scores_prompt = selfcheck.predict(
                    sentences = gpt3_sentences,
                    sampled_passages = LLM_text_samples,
                )
                new_result[LLMtype] = sent_scores_prompt
            
            results.append(new_result)
            labels.append(list(annotation))

with open("outputs/crosscheck_prompt_gpt3{}.json".format("_implicit_pure" if implicit else ""), "w") as fout:
    json.dump({"labels": list(labels), "results": list(results)}, fout, indent=4)
