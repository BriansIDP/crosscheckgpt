import json
import numpy as np
from data.crosscheck.leaderboard import metrics
from scipy.stats import spearmanr, pearsonr


# llm_to_rank = ["beluga", "llama2lm", "falcon", "mistral1", "starling", "openorca"]
llm_to_rank = ["vicuna", "llama2lm", "falcon", "mistral", "mistral1", "starling", "openorca", "llama2", "beluga", "zephyr"]
scores = {}
selfcheckscores = {}
evidence_llm = ["mistral", "llama2", "vicuna", "beluga", "zephyr", "llama"]
# evidence_llm = ["mistral", "llama2", "vicuna", "gpt3"]
refcheckscores = {}
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
    # if llm not in evidence_llm:
    #     local_evidence_llm = evidence_llm[:] + [llm]
    # else:
    local_evidence_llm = evidence_llm[:]
    for paragraph in data:
        localscore = [paragraph[scorellm] for scorellm in local_evidence_llm]
        total_self_score.extend(paragraph[llm])
        for score in zip(*localscore):
            total_score.append(sum(score) / len(score))
    scores[llm] = - sum(total_score) / len(total_score)
    selfcheckscores[llm] = - sum(total_self_score) / len(total_self_score)

    refcheckscore = []
    with open("outputs/crosscheck_prompt_{}_refcheck.json".format(llm)) as fin:
        data = json.load(fin)
    for paragraph in data:
        refcheckscore.extend(paragraph["ref"])
    refcheckscores[llm] = - sum(refcheckscore) / len(refcheckscore)

print("Crosscheck:", scores)
print("Selfcheck:", selfcheckscores)
print("Refcheck:", refcheckscores)

crosscheckscores = np.array([scores[model] for model in llm_to_rank]) * 100
selfcheckscores = np.array([selfcheckscores[model] for model in llm_to_rank]) * 100
refcheckscores = np.array([refcheckscores[model] for model in llm_to_rank]) * 100

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
# all_metrics.extend([selfcheckscores, crosscheckscores])
all_metrics = np.stack(all_metrics)
all_rank = all_metrics.argsort(axis=1).argsort(axis=1)
pcc_matrix = np.zeros((len(all_metrics), len(all_metrics)))
for i, metric in enumerate(all_metrics):
    for j, metric2 in enumerate(all_metrics):
        pcc_matrix[i, j] = spearmanr(metric, metric2)[0]
temp = 0.2
pcc_weight = pcc_matrix * (pcc_matrix > 0.8)
weight = (np.exp(pcc_weight / temp) / np.exp(pcc_weight / temp).sum(axis=0)).diagonal()
weight = np.array([1, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1])
print(weight)
np.save("pcc_matrix.npy", pcc_matrix)
# print(list((pcc_matrix.sum(axis=-1) - 1) / len(all_metrics)))

print("---------TriviaQA---------")
print("SelfCheck PCC: {:.2f}, CrossCheck PCC: {:.2f}, RefCheck PCC: {:.2f}".format(
    pearsonr(selfcheckscores, TriviaQA)[0]*100, pearsonr(crosscheckscores, TriviaQA)[0]*100, pearsonr(refcheckscores, TriviaQA)[0]*100))
print("SelfCheck SRC: {:.2f}, CrossCheck SRC: {:.2f}, RefCheck SRC: {:.2f}".format(
    spearmanr(selfcheckscores, TriviaQA)[0]*100, spearmanr(crosscheckscores, TriviaQA)[0]*100, spearmanr(refcheckscores, TriviaQA)[0]*100))
print("---------TruthQA_MC1---------")
print("SelfCheck PCC: {:.2f}, CrossCheck PCC: {:.2f}, RefCheck PCC: {:.2f}".format(
    pearsonr(selfcheckscores, TruthQA_MC1)[0]*100, pearsonr(crosscheckscores, TruthQA_MC1)[0]*100, pearsonr(refcheckscores, TruthQA_MC1)[0]*100))
print("SelfCheck SRC: {:.2f}, CrossCheck SRC: {:.2f}, RefCheck SRC: {:.2f}".format(
    spearmanr(selfcheckscores, TruthQA_MC1)[0]*100, spearmanr(crosscheckscores, TruthQA_MC1)[0]*100, spearmanr(refcheckscores, TruthQA_MC1)[0]*100))
print("---------TruthQA_MC2---------")
print("SelfCheck PCC: {:.2f}, CrossCheck PCC: {:.2f}, RefCheck PCC: {:.2f}".format(
    pearsonr(selfcheckscores, TruthQA_MC2)[0]*100, pearsonr(crosscheckscores, TruthQA_MC2)[0]*100, pearsonr(refcheckscores, TruthQA_MC2)[0]*100))
print("SelfCheck SRC: {:.2f}, CrossCheck SRC: {:.2f}, RefCheck SRC: {:.2f}".format(
    spearmanr(selfcheckscores, TruthQA_MC2)[0]*100, spearmanr(crosscheckscores, TruthQA_MC2)[0]*100, spearmanr(refcheckscores, TruthQA_MC2)[0]*100))
print("---------Xsum_factKB---------")
print("SelfCheck PCC: {:.2f}, CrossCheck PCC: {:.2f}, RefCheck PCC: {:.2f}".format(
    pearsonr(selfcheckscores, Xsum_factKB)[0]*100, pearsonr(crosscheckscores, Xsum_factKB)[0]*100, pearsonr(refcheckscores, Xsum_factKB)[0]*100))
print("SelfCheck SRC: {:.2f}, CrossCheck SRC: {:.2f}, RefCheck SRC: {:.2f}".format(
    spearmanr(selfcheckscores, Xsum_factKB)[0]*100, spearmanr(crosscheckscores, Xsum_factKB)[0]*100, spearmanr(refcheckscores, Xsum_factKB)[0]*100))
print("---------CNN_DM_BERTP---------")
print("SelfCheck PCC: {:.2f}, CrossCheck PCC: {:.2f}, RefCheck PCC: {:.2f}".format(
    pearsonr(selfcheckscores, CNN_DM_BERTP)[0]*100, pearsonr(crosscheckscores, CNN_DM_BERTP)[0]*100, pearsonr(refcheckscores, CNN_DM_BERTP)[0]*100))
print("SelfCheck SRC: {:.2f}, CrossCheck SRC: {:.2f}, RefCheck SRC: {:.2f}".format(
    spearmanr(selfcheckscores, CNN_DM_BERTP)[0]*100, spearmanr(crosscheckscores, CNN_DM_BERTP)[0]*100, spearmanr(refcheckscores, CNN_DM_BERTP)[0]*100))
print("---------MemoTrap---------")
print("SelfCheck PCC: {:.2f}, CrossCheck PCC: {:.2f}, RefCheck PCC: {:.2f}".format(
    pearsonr(selfcheckscores, MemoTrap)[0]*100, pearsonr(crosscheckscores, MemoTrap)[0]*100, pearsonr(refcheckscores, MemoTrap)[0]*100))
print("SelfCheck SRC: {:.2f}, CrossCheck SRC: {:.2f}, RefCheck SRC: {:.2f}".format(
    spearmanr(selfcheckscores, MemoTrap)[0]*100, spearmanr(crosscheckscores, MemoTrap)[0]*100, spearmanr(refcheckscores, MemoTrap)[0]*100))
print("---------FaithDial---------")
print("SelfCheck PCC: {:.2f}, CrossCheck PCC: {:.2f}, RefCheck PCC: {:.2f}".format(
    pearsonr(selfcheckscores, FaithDial)[0]*100, pearsonr(crosscheckscores, FaithDial)[0]*100, pearsonr(refcheckscores, FaithDial)[0]*100))
print("SelfCheck SRC: {:.2f}, CrossCheck SRC: {:.2f}, RefCheck SRC: {:.2f}".format(
    spearmanr(selfcheckscores, FaithDial)[0]*100, spearmanr(crosscheckscores, FaithDial)[0]*100, spearmanr(refcheckscores, FaithDial)[0]*100))
print("---------HaluQA_Acc---------")
print("SelfCheck PCC: {:.2f}, CrossCheck PCC: {:.2f}, RefCheck PCC: {:.2f}".format(
    pearsonr(selfcheckscores, HaluQA_Acc)[0]*100, pearsonr(crosscheckscores, HaluQA_Acc)[0]*100, pearsonr(refcheckscores, HaluQA_Acc)[0]*100))
print("SelfCheck SRC: {:.2f}, CrossCheck SRC: {:.2f}, RefCheck SRC: {:.2f}".format(
    spearmanr(selfcheckscores, HaluQA_Acc)[0]*100, spearmanr(crosscheckscores, HaluQA_Acc)[0]*100, spearmanr(refcheckscores, HaluQA_Acc)[0]*100))
print("---------HaluSumm---------")
print("SelfCheck PCC: {:.2f}, CrossCheck PCC: {:.2f}, RefCheck PCC: {:.2f}".format(
    pearsonr(selfcheckscores, HaluSumm)[0]*100, pearsonr(crosscheckscores, HaluSumm)[0]*100, pearsonr(refcheckscores, HaluSumm)[0]*100))
print("SelfCheck SRC: {:.2f}, CrossCheck SRC: {:.2f}, RefCheck SRC: {:.2f}".format(
    spearmanr(selfcheckscores, HaluSumm)[0]*100, spearmanr(crosscheckscores, HaluSumm)[0]*100, spearmanr(refcheckscores, HaluSumm)[0]*100))
print("---------HaluDial---------")
print("SelfCheck PCC: {:.2f}, CrossCheck PCC: {:.2f}, RefCheck PCC: {:.2f}".format(
    pearsonr(selfcheckscores, HaluDial)[0]*100, pearsonr(crosscheckscores, HaluDial)[0]*100, pearsonr(refcheckscores, HaluDial)[0]*100))
print("SelfCheck SRC: {:.2f}, CrossCheck SRC: {:.2f}, RefCheck SRC: {:.2f}".format(
    spearmanr(selfcheckscores, HaluDial)[0]*100, spearmanr(crosscheckscores, HaluDial)[0]*100, spearmanr(refcheckscores, HaluDial)[0]*100))

print("=========Total==========")
print("SelfCheck PCC: {:.2f}, CrossCheck PCC: {:.2f}, RefCheck PCC: {:.2f}".format(
    total_selfcheck/len(all_metrics)*100, total_crosscheck/len(all_metrics)*100, total_refcheck/len(all_metrics)*100))
print("SelfCheck SRC: {:.2f}, CrossCheck SRC: {:.2f}, RefCheck SRC: {:.2f}".format(
    total_selfcheck_src/len(all_metrics)*100, total_crosscheck_src/len(all_metrics)*100, total_refcheck_src/len(all_metrics)*100))

all_metrics_ave = np.matmul(np.transpose(all_metrics), weight)
all_ranks_ave = np.matmul(np.transpose(all_rank), weight)
print("SelfCheck PCC-Mean: {:.2f}, CrossCheck PCC-Mean: {:.2f}, RefCheck PCC-Mean: {:.2f}".format(
    spearmanr(selfcheckscores, all_metrics_ave)[0]*100, spearmanr(crosscheckscores, all_metrics_ave)[0]*100, spearmanr(refcheckscores, all_metrics_ave)[0]*100))
print("SelfCheck SRC-Mean: {:.2f}, CrossCheck SRC-Mean: {:.2f}, RefCheck SRC-Mean: {:.2f}".format(
    spearmanr(selfcheckscores, all_ranks_ave)[0]*100, spearmanr(crosscheckscores, all_ranks_ave)[0]*100, spearmanr(refcheckscores, all_ranks_ave)[0]*100))
