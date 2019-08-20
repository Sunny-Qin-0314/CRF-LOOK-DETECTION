import os
import pickle
import pycrfsuite
from constants import WITHOBJ
from crfsuite_data import prepare_data
from reporting import StatsManager, pretty_print_report, pretty_rl_table


def post_processing(pred):
    for i in range(len(pred)):
        current = pred[i]
        if i > 0:
            prev = pred[i-1]
            if prev =="O" and current.startswith("I-"):
                pred[i] = current.replace("I-","B-")
    return pred


"""
Do evaluation on new test set, video level cross validation

"""

with open(os.path.join("data/out", "new_test.pkl"), "rb") as f:
    validation = pickle.load(f)

support_threshold = 0
stats = StatsManager(support_threshold)

for i, data in enumerate(validation):
    tagger = pycrfsuite.Tagger()
    tagger.open('exp_{}'.format(i))
    # print(len(data))
    y_pred = []
    y_num_pred =[]
    y_true = []
    y_num_true = []
    for features, ylabel in prepare_data(data):
        pred = post_processing(tagger.tag(features))
        y_pred.append(pred)
        y_true.append(ylabel)

    # if i == 4:
    #     print(y_true[0][62:100], y_pred[0][62:100])
    stats.append_report(y_true, y_pred)

# data = validation[4]
#
# tagger = pycrfsuite.Tagger()
# tagger.open('exp_{}'.format(4))
# y_pred = []
# y_num_pred =[]
# y_true = []
# y_num_true = []
# for features, ylabel in prepare_data(data):
#     pred = post_processing(tagger.tag(features))
#     y_pred.append(pred)
#     y_true.append(ylabel)
#
# # if i == 4:
# #     print(y_true[0][62:100], y_pred[0][62:100])
# stats.append_report(y_true, y_pred)
#

# """
# Do evaluation on a seperate test set, after do evaluation on chunk level cross validation
# """

# with open(os.path.join("data/out", "test.pkl"), "rb") as f:
#     test = pickle.load(f)
#
# test_support_threshold = 0
# test_stats = StatsManager(support_threshold)
#
# print(len(test))
# for j, data in enumerate(test):
#     for i in range(6):
#         tagger = pycrfsuite.Tagger()
#         tagger.open('exp_{}'.format(i+5))
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
# #




print("Multi-class Classification Validation Report Mean(Std)")
report, summary = stats.summarize()
pretty_print_report(report)

# print("Multi-class Classification Testing Report Mean(Std)")
# report, summary = test_stats.summarize()
# pretty_print_report(report)

# print("Run-length Report")
# rl_report = stats.runlength_report()
# pretty_rl_table(rl_report)


"""
To see the details of the model, such as the top likely transitions and top features

"""
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

# print("Top positive:")
# print_state_features(Counter(info.state_features).most_common(20))
#
# print("\nTop negative:")
# print_state_features(Counter(info.state_features).most_common()[-20:])
