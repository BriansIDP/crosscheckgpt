import json
import torch
import numpy as np
from data.crosscheck.leaderboard import metrics
from scipy.stats import spearmanr, pearsonr


llm_to_rank = ["beluga", "mistral", "vicuna", "llama2", "falcon", "llama2lm", "mistral1", "starling", "openorca", "zephyr"]
scores = {}
selfcheckscores = {}
evidence_llm = ["mistral", "llama2", "vicuna", "zephyr", "beluga", "starling", "openorca", "llama2lm"]

caliberation = 0.1

selfscores = {}
for llm in llm_to_rank + evidence_llm:
    selfscores[llm] = []
    with open("outputs/crosscheck_prompt_{}.json".format(llm)) as fin:
        data = json.load(fin)
    for paragraph in data:
        selfscores[llm].extend(paragraph[llm])
    selfscores[llm] = sum(selfscores[llm]) / len(selfscores[llm]) / 20
# Per passage weighting
selfscores_passages = {}
for llm in llm_to_rank:
    selfscores_passages[llm] = []
    with open("outputs/crosscheck_prompt_{}.json".format(llm)) as fin:
        data = json.load(fin)
    for paragraph in data:
        selfscores_passages[llm].append(sum(paragraph[llm]) / len(paragraph[llm]) / 20)

with open("outputs/crosscheck_implicit_cot_all.json") as fin:
    crosscheck_imp_raw = json.load(fin)
crosscheckscores_imp = {}
selfscores_array = - torch.tensor([selfscores[ellm] for ellm in evidence_llm])
weight_array = torch.softmax(selfscores_array/caliberation, dim=-1) # * len(local_evidence_llm)
crossweights = {ellm: weight_array[i].item() for i, ellm in enumerate(evidence_llm)}
for model in llm_to_rank:
    passage_level = []
    if model in crosscheck_imp_raw:
        for passage in crosscheck_imp_raw[model]:
            imp_scores = [sum(passage[llm])/len(passage[llm]) * crossweights[llm] for llm in evidence_llm if llm in passage]
            imp_scores = sum(imp_scores) # / len(imp_scores)
            passage_level.append(imp_scores)
        crosscheckscores_imp[model] = passage_level

refcheckscores = {}
for llm in llm_to_rank:
    if llm == "gpt3":
        with open("outputs/crosscheck_prompt.json") as fin:
            data = json.load(fin)
        data = data["results"]
    else:
        with open("outputs/crosscheck_prompt_{}.json".format(llm)) as fin:
            data = json.load(fin)

    if llm not in evidence_llm:
        local_evidence_llm = evidence_llm[:] + [llm]
    else:
        local_evidence_llm = evidence_llm[:]

    # selfscores_array = - torch.tensor([selfscores_passages[ellm] for ellm in local_evidence_llm])
    selfscores_array = - torch.tensor([selfscores[ellm] for ellm in local_evidence_llm])
    weight_array = torch.softmax(selfscores_array/caliberation, dim=-1) # * len(local_evidence_llm)
    print("Weights for {}: ".format(llm), weight_array.tolist())
    crossweights = {ellm: weight_array[i].item() for i, ellm in enumerate(local_evidence_llm)}

    total_score = []
    total_self_score = []
    for i, paragraph in enumerate(data):
        total_self_score.append(sum(paragraph[llm]) / len(paragraph[llm]))
        # Add weight
        # weight_array = torch.softmax(selfscores_array[:, i]/caliberation, dim=-1)
        # crossweights = {ellm: weight_array[i].item() for i, ellm in enumerate(local_evidence_llm)}

        for scorellm in local_evidence_llm:
            paragraph[scorellm] = [score * crossweights[scorellm] for score in paragraph[scorellm]]

        paragraph_score = []
        localscore = [paragraph[scorellm] for scorellm in local_evidence_llm]
        for score in zip(*localscore):
            # paragraph_score.append(sum(score) / len(score))
            paragraph_score.append(sum(score))
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

total_cross = []
total_self = []
total_chair = []
total_imp = []
total_cross_points = []
total_self_points = []
total_ref_points = []
total_imp_points = []

for llm in llm_to_rank:
    print(llm)
    print("SelfCheck vs. RefCheck:", pearsonr(selfcheckscores[llm], refcheckscores[llm])[0])
    print("CrossCheck vs. RefCheck:", pearsonr(scores[llm], refcheckscores[llm])[0])
    print("SelfCheck vs. RefCheck:", ((selfcheckscores[llm] - refcheckscores[llm] * 20) ** 2).mean())
    print("CrossCheck vs. RefCheck:", ((scores[llm] - refcheckscores[llm] * 20) ** 2).mean())
    if llm in crosscheckscores_imp:
        print("CrossCheck_implicit vs. RefCheck:", pearsonr(crosscheckscores_imp[llm], refcheckscores[llm])[0])
        total_imp.append(pearsonr(crosscheckscores_imp[llm], refcheckscores[llm])[0])
        total_imp_points.extend(crosscheckscores_imp[llm])
    total_cross.append(pearsonr(scores[llm], refcheckscores[llm])[0])
    total_self.append(pearsonr(selfcheckscores[llm], refcheckscores[llm])[0])
    total_cross_points.extend(scores[llm])
    total_self_points.extend(selfcheckscores[llm])
    total_ref_points.extend(refcheckscores[llm])

total_cross_points = np.array(total_cross_points)
total_self_points = np.array(total_self_points)
total_ref_points = np.array(total_ref_points)
total_imp_points = np.array(total_imp_points)
print("==========Overall==========")
print("SelfCheck vs. Chair PCC:", pearsonr(total_self_points, total_ref_points)[0])
print("CrossCheck vs. Chair PCC:", pearsonr(total_cross_points, total_ref_points)[0])
print("CrossCheck_implicit vs. RefCheck:", pearsonr(total_imp_points, total_ref_points)[0])
