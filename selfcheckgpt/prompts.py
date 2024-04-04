questionpromptcot = """You are given the following sentence about {person}:
{sentence}
List any factual errors in the sentence."""

questionpromptcot2 = """You are given the following sentence about {person}:
{sentence}
The following is an analysis of possible factual errors in this sentence:
{analysis}
Based on the analysis, determine if the sentence contains any factual errors. Answer YES if it has errors, and NO if it does not. Your answer should be in the following format:
{{"Answer": <YES_or_NO>}}"""

questionprompt = """You are given the following sentence about {person}:
{sentence}
Determine if the sentence contains any factual errors. Answer YES if it has errors, and NO if it does not. Your answer should be in the following format:
{{"Answer": <YES_or_NO>}}"""

mistral = "[INST] {} [/INST]"
llama2 = "[INST] {} [/INST]"
vicuna = "USER: {}\nASSISTANT:"
beluga = "<s> ### User:\n{}\n\n###Assistant:\n"
zephyr = "<|user|>\n{}\n\n<|assistant|>\n"
starling = "GPT4 Correct User: {}<|end_of_turn|>GPT4 Correct Assistant:"
llama2lm = "{}\n"
falcon = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:"
openorca = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
