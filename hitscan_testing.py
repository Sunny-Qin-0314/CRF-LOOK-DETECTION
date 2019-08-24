import os
import pickle
import pycrfsuite

from crfsuite_data import prepare_data
from reporting import StatsManager, pretty_print_report, pretty_rl_table

with open(os.path.join("data/out", "new_test.pkl"), "rb") as f:
    validation = pickle.load(f)

def ftag(features):
    pred = []
    for f in features:
        p = "O"
        # if f['is_card']:
        #     p = "card"
        if f['is_dice']:
            p="I-dice"
        elif f['is_map']:
            p="I-map"
        elif f['is_ball']:
            p="I-ball"
        elif f['is_face']:
            p="I-face"
        elif f['is_key']:
            p="I-key"
        pred.append(p)
    return pred

support_threshold = 0
stats = StatsManager(support_threshold)

for i, data in enumerate(validation):
    y_pred = []
    y_true = []
    for features, ylabel in prepare_data(data):
        y_pred.append(ftag(features))
        y_true.append(ylabel)
# print(y_true[0][0:50])
# print("predict:", y_pred[0][0:50])
    stats.append_report(y_true, y_pred)

print("Multi-class Classification Report Mean(Std)")
report, summary = stats.summarize()
# pretty_print_report(report)

# print("Run-length Report")
# rl_report = stats.runlength_report()
# pretty_rl_table(rl_report)
