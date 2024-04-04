import json
import numpy as np
from data.crosscheck.leaderboard import metrics
from scipy.stats import spearmanr, pearsonr


# llm_to_rank = ["beluga", "mistral", "vicuna", "llama2", "zephyr"]

llm_to_rank = ["beluga", "mistral", "vicuna", "llama2", "falcon", "llama2lm", "mistral1", "starling", "openorca", "zephyr"]
scores = {}
selfcheckscores = {}
evidence_llm = ["mistral", "llama2", "vicuna", "llama", "beluga", "zephyr"]
# evidence_llm = ["mistral", "llama2", "vicuna"]
refcheckscores = {}
for llm in llm_to_rank:
    if llm == "gpt3":
        with open("outputs/crosscheck_prompt.json") as fin:
            data = json.load(fin)
        data = data["results"]
    else:
        with open("outputs/crosscheck_prompt_{}.json".format(llm)) as fin:
            data = json.load(fin)
    total_score = []
    total_self_score = []
    for paragraph in data:
        paragraph_score = []
        localscore = [paragraph[scorellm] for scorellm in evidence_llm]
        total_self_score.append(sum(paragraph[llm]) / len(paragraph[llm]))
        for score in zip(*localscore):
            paragraph_score.append(sum(score) / len(score))
        total_score.append(sum(paragraph_score)/len(paragraph_score))
    scores[llm] = np.array(total_score)
    selfcheckscores[llm] = np.array(total_self_score)
    # scores[llm] = - sum(total_score) / len(total_score)
    # selfcheckscores[llm] = - sum(total_self_score) / len(total_self_score)

    refcheckscore = []
    with open("outputs/crosscheck_prompt_{}_refcheck.json".format(llm)) as fin:
        data = json.load(fin)
    for paragraph in data:
        refcheckscore.append(sum(paragraph["ref"]) / len(paragraph["ref"]))
    # refcheckscores[llm] = - sum(refcheckscore) / len(refcheckscore)
    refcheckscores[llm] = np.array(refcheckscore)

for llm in llm_to_rank:
    print(llm)
    print("SelfCheck vs. RefCheck:", pearsonr(selfcheckscores[llm], refcheckscores[llm])[0])
    print("CrossCheck vs. RefCheck:", pearsonr(scores[llm], refcheckscores[llm])[0])
    print("SelfCheck vs. RefCheck:", ((selfcheckscores[llm] - refcheckscores[llm] * 20) ** 2).mean())
    print("CrossCheck vs. RefCheck:", ((scores[llm] - refcheckscores[llm] * 20) ** 2).mean())


closedomain = ["TriviaQA", "Xsum_factKB", "CNN_DM_BERTP"]
opendomain = ["TruthQA_MC1", "TruthQA_MC2", "MemoTrap", "HaluQA_Acc"]
