import spacy
# import bert_score
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union
import json
from transformers import logging
logging.set_verbosity_error()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import LongformerTokenizer, LongformerForMultipleChoice, LongformerForSequenceClassification
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
from crosscheckgpt.utils import MQAGConfig, expand_list1, expand_list2, NLIConfig, LLMPromptConfig
from crosscheckgpt.prompts import questionprompt, questionpromptcot, questionpromptcot2

# ---------------------------------------------------------------------------------------- #
# Functions for counting
def method_simple_counting(
    prob,
    u_score,
    prob_s,
    u_score_s,
    num_samples,
    AT,
):
    """
    simple counting method score => count_mismatch / (count_match + count_mismatch)
    :return score: 'inconsistency' score
    """
    # bad questions, i.e. not answerable given the passage
    if u_score < AT:
        return 0.5
    a_DT = np.argmax(prob)
    count_good_sample, count_match = 0, 0
    for s in range(num_samples):
        if u_score_s[s] >= AT:
            count_good_sample += 1
            a_S = np.argmax(prob_s[s])
            if a_DT == a_S:
                count_match += 1
    if count_good_sample == 0:
        score = 0.5
    else:
        score = (count_good_sample-count_match) / count_good_sample
    return score

def method_vanilla_bayes(
    prob,
    u_score,
    prob_s,
    u_score_s,
    num_samples,
    beta1, beta2, AT,
):
    """
    (vanilla) bayes method score: compute P(sentence is non-factual | count_match, count_mismatch)
    :return score: 'inconsistency' score
    """
    if u_score < AT:
        return 0.5
    a_DT = np.argmax(prob)
    count_match, count_mismatch = 0, 0
    for s in range(num_samples):
        if u_score_s[s] >= AT:
            a_S = np.argmax(prob_s[s])
            if a_DT == a_S:
                count_match += 1
            else:
                count_mismatch += 1
    gamma1 = beta2 / (1.0-beta1)
    gamma2 = beta1 / (1.0-beta2)
    score = (gamma2**count_mismatch) / ((gamma1**count_match) + (gamma2**count_mismatch))
    return score

def method_bayes_with_alpha(
    prob,
    u_score,
    prob_s,
    u_score_s,
    num_samples,
    beta1, beta2,
):
    """
    bayes method (with answerability score, i.e. soft-counting) score
    :return score: 'inconsistency' score
    """
    a_DT = np.argmax(prob)
    count_match, count_mismatch = 0, 0
    for s in range(num_samples):
        ans_score = u_score_s[s]
        a_S = np.argmax(prob_s[s])
        if a_DT == a_S:
            count_match += ans_score
        else:
            count_mismatch += ans_score
    gamma1 = beta2 / (1.0-beta1)
    gamma2 = beta1 / (1.0-beta2)
    score = (gamma2**count_mismatch) / ((gamma1**count_match) + (gamma2**count_mismatch))
    return score

def answerability_scoring(
    u_model,
    u_tokenizer,
    question,
    context,
    max_length,
    device,
):
    """
    :return prob: prob -> 0.0 means unanswerable, prob -> 1.0 means answerable
    """
    input_text = question + ' ' + u_tokenizer.sep_token + ' ' + context
    inputs = u_tokenizer(input_text, max_length=max_length, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)
    logits = u_model(**inputs).logits
    logits = logits.squeeze(-1)
    prob = torch.sigmoid(logits).item()
    return prob


class CrossCheckNLI:
    """
    CrossCheckGPT (NLI variant): Checking LLM's text against its own sampled texts via DeBERTa-v3 finetuned to Multi-NLI
    """
    def __init__(
        self,
        nli_model: str = None,
        device = None,
        cache_dir: str = None,
        threeway: bool = False,
    ):
        nli_model = nli_model if nli_model is not None else NLIConfig.nli_model
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(nli_model)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(nli_model, cache_dir=cache_dir)
        if threeway:
            self.model.classifier = torch.nn.Linear(in_features=1024, out_features=3)
            state_dict = torch.load("/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/LLMknowledge/ckpt/nli_threeway_cls.pt")
            self.model.classifier.weight.data = state_dict["classifier.weight"]
            self.model.classifier.bias.data = state_dict["classifier.bias"]
        self.model.eval()
        if device is None:
            device = torch.device("cpu")
        self.model.to(device)
        self.device = device
        print("CrossCheck-NLI initialized to device", device)

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :return sent_scores: sentence-level score which is P(condict|sentence, sample)
        note that we normalize the probability on "entailment" or "contradiction" classes only
        and the score is the probability of the "contradiction" class
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores_per_sentence = []
        for sent_i, sentence in enumerate(sentences):
            sample_pairs = [(sentence, sample) for sample in sampled_passages]
            inputs = self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=sample_pairs,
                add_special_tokens=True, padding="longest",
                truncation=True, return_tensors="pt",
                return_token_type_ids=True, return_attention_mask=True,
            )
            inputs = inputs.to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prob_ = probs[:, 1].mean().item()
            scores_per_sentence.append(prob_)
        return scores_per_sentence


class CrossCheckLLMPrompt:
    """
    CrossCheckGPT (LLM Prompt): Checking LLM's text against its own sampled texts via open-source LLM prompting
    """
    def __init__(
        self,
        model: str = None,
        device = None,
        cache_dir: str = None,
    ):
        model = model if model is not None else LLMPromptConfig.model
        self.tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", cache_dir=cache_dir)
        self.model.eval()
        if device is None:
            device = torch.device("cpu")
        self.model.to(device)
        self.device = device
        self.prompt_template = """Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: """
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()
        print(f"CrossCheck-LLMPrompt ({model}) initialized to device {device}")

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False,
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :param verson: bool -- if True tqdm progress bar will be shown
        :return sent_scores: sentence-level scores
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose
        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(sampled_passages):
                
                # this seems to improve performance when using the simple prompt template
                sample = sample.replace("\n", " ") 

                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                generate_ids = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=5,
                    do_sample=False, # hf's default for Llama2 is True
                )
                output_text = self.tokenizer.batch_decode(
                    generate_ids[:, inputs.input_ids.size(1):], skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                generate_text = output_text.replace(prompt, "")
                score_ = self.text_postprocessing(generate_text)
                scores[sent_i, sample_i] = score_
        # scores_per_sentence = scores.mean(axis=-1)
        scores_per_sentence = scores.sum(axis=-1)
        return list(scores_per_sentence)

    def text_postprocessing(
        self,
        text,
    ):
        """
        To map from generated text to score
        Yes -> 0.0
        No  -> 1.0
        everything else -> 0.5
        """
        # tested on Llama-2-chat (7B, 13B) --- this code has 100% coverage on wikibio gpt3 generated data
        # however it may not work with other datasets, or LLMs
        text = text.lower().strip()
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]

class CrossCheckQuestionsLLMPrompt:
    """
    CrossCheckGPT (LLM Prompt): Checking LLM's text against its own sampled texts via open-source LLM prompting
    """
    def __init__(
        self,
        model: str = None,
        device = None,
        cache_dir: str = None,
        cot: bool = False,
        prompt_template: str = None,
    ):
        model = model if model is not None else LLMPromptConfig.model
        self.tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", cache_dir=cache_dir)
        self.model.eval()
        if device is None:
            device = torch.device("cpu")
        self.model.to(device)
        self.device = device
        if cot:
            self.prompt_template_cot = prompt_template.format(questionpromptcot)
            self.prompt_template_cot2 = prompt_template.format(questionpromptcot2)
        else:
            self.prompt_template = prompt_template.format(questionprompt)
        self.text_mapping = {'yes': 1.0, 'no': 0.0, 'n/a': 0.5}
        self.cot = cot
        self.not_defined_text = set()
        print(f"CrossCheck-LLMPrompt ({model}) initialized to device {device}")

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        person: str,
        verbose: bool = False,
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :param verson: bool -- if True tqdm progress bar will be shown
        :return sent_scores: sentence-level scores
        """
        num_sentences = len(sentences)
        scores = []
        for sent_i in range(num_sentences):
            sentence = sentences[sent_i]
            if self.cot:
                prompt_1 = self.prompt_template_cot.format(**locals())
                inputs = self.tokenizer(prompt_1, return_tensors="pt").to(self.device)
                generate_ids = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=128,
                    do_sample=False, # hf's default for Llama2 is True
                )
                analysis = self.tokenizer.batch_decode(
                    generate_ids[:, inputs.input_ids.size(1):], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                prompt_2 = self.prompt_template_cot2.format(**locals())
                inputs = self.tokenizer(prompt_2, return_tensors="pt").to(self.device)
                generate_ids = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=8,
                    do_sample=False, # hf's default for Llama2 is True
                )
                generate_text = self.tokenizer.batch_decode(
                    generate_ids[:, inputs.input_ids.size(1):], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
            else:
                prompt = self.prompt_template.format(**locals())
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                generate_ids = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=8,
                    do_sample=False, # hf's default for Llama2 is True
                )
                generate_text = self.tokenizer.batch_decode(
                    generate_ids[:, inputs.input_ids.size(1):], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
            score_ = self.text_postprocessing(generate_text)
            scores.append(score_)
        return scores

    def text_postprocessing(
        self,
        text,
    ):
        """
        To map from generated text to score
        Yes -> 1.0
        No  -> 0.0
        everything else -> 0.5
        """
        # tested on Llama-2-chat (7B, 13B) --- this code has 100% coverage on wikibio gpt3 generated data
        # however it may not work with other datasets, or LLMs
        text = text.lower()
        text = text.strip().lstrip()
        if "yes" in text and "no" not in text:
            text = 'yes'
        elif "no" in text and "yes" not in text:
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]
