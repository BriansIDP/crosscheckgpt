import json
import random
import numpy as np
import torch
from data.crosscheck.leaderboard import metrics
from scipy.stats import spearmanr, pearsonr


llm_to_rank = ["vicuna", "llama2lm", "falcon", "mistral", "mistral1", "starling", "openorca", "llama2", "beluga", "zephyr"]
scores = {}
selfcheckscores = {}
evidence_llm = ["mistral", "llama2", "vicuna", "zephyr", "beluga", "starling", "openorca", "llama2lm"]
caliberation = 0.1

# Overall weighting
selfscores = {}
for llm in llm_to_rank:
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

refcheckscores = {}
crosscheckdetail = []
selfcheckdetail = []
for llm in llm_to_rank:
    if llm == "gpt3":
        with open("outputs/crosscheck_prompt_gpt3.json") as fin:
            data = json.load(fin)
        data = data["results"]
    else:
        with open("outputs/crosscheck_prompt_{}.json".format(llm)) as fin:
            data = json.load(fin)

    total_score = []
    total_self_score = []
    if llm not in evidence_llm:
        local_evidence_llm = evidence_llm[:] + [llm]
    else:
        local_evidence_llm = evidence_llm[:]

    # selfscores_array = - torch.tensor([selfscores_passages[ellm] for ellm in local_evidence_llm])
    selfscores_array = - torch.tensor([selfscores[ellm] for ellm in local_evidence_llm])
    weight_array = torch.softmax(selfscores_array/caliberation, dim=-1) # * len(local_evidence_llm)
    # print("Weights for {}: ".format(llm), weight_array.tolist())
    crossweights = {ellm: weight_array[i].item() for i, ellm in enumerate(local_evidence_llm)}

    passage_level_cross = []
    passage_level_self = []
    for i, paragraph in enumerate(data):
        # Uncomment for per-passage weight
        # weight_array = torch.softmax(selfscores_array[:, i]/caliberation, dim=-1)
        # crossweights = {ellm: weight_array[i].item() for i, ellm in enumerate(local_evidence_llm)}
        passage_level_self.append(sum(paragraph[llm])/len(paragraph[llm])/20)
        for scorellm in local_evidence_llm:
            paragraph[scorellm] = [score * crossweights[scorellm] for score in paragraph[scorellm]]

        localscore = [paragraph[scorellm] for scorellm in local_evidence_llm]

        total_self_score.extend(paragraph[llm])
        passage_level_cross.append(np.array(localscore).sum(axis=0).mean() / 20)
        for score in zip(*localscore):
            total_score.append(sum(score))
    scores[llm] = - sum(total_score) / len(total_score) / 20
    crosscheckdetail.append(passage_level_cross)
    selfcheckdetail.append(passage_level_self)
    selfcheckscores[llm] = - sum(total_self_score) / len(total_self_score)

    refcheckscore = []
    with open("outputs/crosscheck_prompt_{}_refcheck.json".format(llm)) as fin:
        data = json.load(fin)
    for paragraph in data:
        refcheckscore.extend(paragraph["ref"])
    refcheckscores[llm] = - sum(refcheckscore) / len(refcheckscore)

selfcheckscores = selfscores
crosscheckscores = np.array([scores[model] for model in llm_to_rank]) * 100
selfcheckscores = np.array([selfcheckscores[model] for model in llm_to_rank]) * 100
refcheckscores = np.array([refcheckscores[model] for model in llm_to_rank]) * 100
crosscheckdetail = np.array(crosscheckdetail)
selfcheckdetail = np.array(selfcheckdetail)
# print("="*89)
# print("Crosscheck Scores:", [s for m, s in scores.items()])
# print("Selfcheck Scores:", selfcheckscores)
# print("Refcheck Scores:", refcheckscores)
# print("="*89)

TriviaQA = np.array([metrics[model]["TriviaQA"] for model in llm_to_rank])
TruthQA_MC1 = np.array([metrics[model]["TruthQA_MC1"] for model in llm_to_rank])
TruthQA_MC2 = np.array([metrics[model]["TruthQA_MC2"] for model in llm_to_rank])
Xsum_factKB = np.array([metrics[model]["Xsum_factKB"] for model in llm_to_rank])
CNN_DM_BERTP = np.array([metrics[model]["CNN_DM_BERTP"] for model in llm_to_rank])
MemoTrap = np.array([metrics[model]["MemoTrap"] for model in llm_to_rank])
FaithDial = np.array([metrics[model]["FaithDial"] for model in llm_to_rank])
HaluQA_Acc = np.array([metrics[model]["HaluQA_Acc"] for model in llm_to_rank])
HaluSumm = np.array([metrics[model]["HaluSumm"] for model in llm_to_rank])
HaluDial = np.array([metrics[model]["HaluDial"] for model in llm_to_rank])

total_selfcheck = 0
total_crosscheck = 0
total_refcheck = 0
total_selfcheck_src = 0
total_crosscheck_src = 0
total_refcheck_src = 0
all_metrics = [TriviaQA, TruthQA_MC1, TruthQA_MC2, Xsum_factKB, CNN_DM_BERTP, MemoTrap, FaithDial, HaluQA_Acc, HaluSumm, HaluDial]
for metric in all_metrics:
    total_selfcheck += pearsonr(selfcheckscores, metric)[0]
    total_crosscheck += pearsonr(crosscheckscores, metric)[0]
    total_refcheck += pearsonr(refcheckscores, metric)[0]
    total_selfcheck_src += spearmanr(selfcheckscores, metric)[0]
    total_crosscheck_src += spearmanr(crosscheckscores, metric)[0]
    total_refcheck_src += spearmanr(refcheckscores, metric)[0]
all_metrics.extend([selfcheckscores, crosscheckscores])
all_metrics = np.stack(all_metrics)
all_rank = all_metrics.argsort(axis=1).argsort(axis=1)
pcc_matrix = np.zeros((len(all_metrics), len(all_metrics)))
for i, metric in enumerate(all_metrics):
    for j, metric2 in enumerate(all_metrics):
        pcc_matrix[i, j] = spearmanr(metric, metric2)[0]
weight = np.array([1, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 0, 0])
np.save("pcc_matrix.npy", pcc_matrix)

all_metrics_ave = np.matmul(np.transpose(all_metrics), weight)
all_ranks_ave = np.matmul(np.transpose(all_rank), weight)
# print("Overall ranking:", list(all_ranks_ave))
print("="*89)
print("SelfCheck system-level: {:.2f}, CrossCheck system-level: {:.2f}".format(
    spearmanr(selfcheckscores, all_ranks_ave)[0]*100, spearmanr(crosscheckscores, all_ranks_ave)[0]*100))

# Implicit
with open("outputs/crosscheck_implicit_cot_all.json") as fin:
    crosscheck_imp_raw = json.load(fin)
crosscheckscores_imp = []
selfscores_array = - torch.tensor([selfscores[ellm] for ellm in evidence_llm])
weight_array = torch.softmax(selfscores_array/caliberation, dim=-1)
crossweights = {ellm: weight_array[i].item() for i, ellm in enumerate(local_evidence_llm)}
for model in llm_to_rank:
    passage_level = []
    for passage in crosscheck_imp_raw[model]:
        scores = [sum(passage[llm])/len(passage[llm])*crossweights[llm] for llm in evidence_llm if llm in passage]
        scores = sum(scores) # / len(scores)
        passage_level.append(scores)
    crosscheckscores_imp.append(-sum(passage_level) / len(passage_level))
print("="*89)
print("CrossCheck Implicit system-level: {:.2f}".format(spearmanr(crosscheckscores_imp, all_ranks_ave)[0]*100))

hit = 0
random_id_list = [i for i in range(238)]
crossscores_total = []
selfscores_total = []
npoints = 1
# for i in range(200):
for idx in range(30):
    random_ids = [n for n in range(idx, idx+8)] # random.sample(random_id_list, npoints)
    crossscores = crosscheckdetail[:, random_ids]
    crossscores = crossscores.sum(axis=-1) / npoints
    crossscores_total.append(crossscores)
    selfscores = selfcheckdetail[:, random_ids]
    selfscores = selfscores.sum(axis=-1) / npoints
    selfscores_total.append(selfscores)
    src_cross = spearmanr(-crossscores, all_ranks_ave)[0]
    src_self = spearmanr(-selfscores, all_ranks_ave)[0]
    if src_cross > src_self:
        hit += 1
print("="*89)
print("Success Rate:", hit/30)
print("="*89)
crossscores_total = np.concatenate(crossscores_total, axis=-1)
selfscores_total = np.concatenate(selfscores_total, axis=-1)
all_ranks_ave_total = np.concatenate([all_ranks_ave for _ in range(10)], axis=-1)
# np.save("outputs/crosscheck_rankings.npy", crossscores_total)
# np.save("outputs/selfcheck_rankings.npy", selfscores_total)
# np.save("outputs/overall_rankings.npy", all_ranks_ave_total)
