import json


with open("sentences_vicuna13b.json") as fin:
    data = json.load(fin)

newutts = []
for uttid, utt in data.items():
    newutt = [u[0] for u in utt]
    print(len(utt))
    data[uttid] = newutt

with open("sentences_vicuna13b.json", "w") as fout:
    json.dump(data, fout, indent=4)
