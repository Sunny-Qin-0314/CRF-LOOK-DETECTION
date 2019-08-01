import os
import pickle
import pycrfsuite

from crfsuite_data import prepare_data
from reporting import StatsManager, pretty_print_report, pretty_rl_table

"""
Do evaluation on validation set

"""

with open(os.path.join("data/out", "validation.pkl"), "rb") as f:
    validation = pickle.load(f)

support_threshold = 100
stats = StatsManager(support_threshold)

# print(len(validation))
for i, data in enumerate(validation):
    tagger = pycrfsuite.Tagger()
    tagger.open('exp_{}'.format(i))

    y_pred = []
    y_true = []
    for features, ylabel in prepare_data(data):
        y_pred.append(tagger.tag(features))
        y_true.append(ylabel)

    stats.append_report(y_true, y_pred)


# """
# Do evaluation on test set 
# """
#
# with open(os.path.join("data/out", "test.pkl"), "rb") as f:
#     test = pickle.load(f)
#
# test_support_threshold = 0
# test_stats = StatsManager(support_threshold)
#
# print(len(test))
# for j, data in enumerate(test):
#     for i in range(5):
#         tagger = pycrfsuite.Tagger()
#         tagger.open('exp_{}'.format(i))
#
#         test_y_pred = []
#         test_y_true = []
#         test_feature_total = []
#
#         temp = prepare_data(data)
#         # print(temp[13][340:350])
#         for features, ylabel in temp:
#
#                   test_feature_total.append(features)  #900*n
#
#                   test_y_true.append(ylabel)#900*n
#
#                   test_y_pred.append(tagger.tag(features))
#         test_stats.append_report(test_y_true, test_y_pred)





print("Multi-class Classification Validation Report Mean(Std)")
report, summary = stats.summarize()
pretty_print_report(report)

# print("Multi-class Classification Testing Report Mean(Std)")
# report, summary = test_stats.summarize()
# pretty_print_report(report)
#
# print("Run-length Report")
# rl_report = stats.runlength_report()
# pretty_rl_table(rl_report)

from collections import Counter
info = tagger.info()

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

# print("Top likely transitions:")
# print_transitions(Counter(info.transitions).most_common(15))
#
# print("\nTop unlikely transitions:")
# print_transitions(Counter(info.transitions).most_common()[-15:])

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))

print("Top positive:")
print_state_features(Counter(info.state_features).most_common(20))

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-20:])
