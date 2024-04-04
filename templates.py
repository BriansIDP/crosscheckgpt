mistral = [
    "[INST] Generate a passage about {}. [/INST]",
    "[INST] Give me information about {} in a paragraph. [/INST]",
    "[INST] Can you tell me something about {}? [/INST]",
]
llama2 = [
    "[INST] Generate a passage about {}. [/INST]",
    "[INST] Give me information about {} in a paragraph. [/INST]",
    "[INST] Can you tell me something about {}? [/INST]",
]
vicuna = [
    "USER: Generate a passage about {}. ASSISTANT:",
    "USER: Give me information about {} in a paragraph. ASSISTANT:",
    "USER: Can you tell me something about {}? ASSISTANT:",
]
vicuna13b = [
    "USER: Generate a passage about {}. ASSISTANT:",
    "USER: Give me information about {} in a paragraph. ASSISTANT:",
    "USER: Can you tell me something about {}? ASSISTANT:",
]
llama2lm = [
    "This is a Wikipedia passage about {}:\n",
    "The following paragraph summarises information about {}:\n",
]
starling = [
    "GPT4 Correct User: Generate a passage about {}.<|end_of_turn|>GPT4 Correct Assistant:",
    "GPT4 Correct User: Give me information about {} in a paragraph.<|end_of_turn|>GPT4 Correct Assistant:",
]
bloom = [
    "This is a paragraph about {} from wikipedia:\n",
    "The following paragraph summarises information about {}:\n",
]
falcon = [
    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nGenerate a passage about {}.\n\n### Input:",
    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nGive me information about {} in a paragraph.\n\n### Input:"
]
openorca = [
    "<|im_start|>user\nGenerate a passage about {}.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nGive me information about {} in a paragraph.<|im_end|>\n<|im_start|>assistant\n"
]
