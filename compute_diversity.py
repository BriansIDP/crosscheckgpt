import json


div_model = "avsalmonn_av"

with open("outputs/diversity_{}_selfcheck.json".format(div_model)) as fin:
    data = json.load(fin)

total_score = 0
total_count = 0
for pid, scores in data.items():
    total_score += sum(scores) / len(scores)
    total_count += 1

print(total_score / total_count)
