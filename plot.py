"""
Use this file to see one specific model performance, especially how the pred covers gt.
It can also plot the feature(within_bbox) on the gt.
Also, confusion matrix can be plotted

"""
# #
# import pycrfsuite
# import os
# import pickle
# import crfsuite_data
# from crfsuite_data import prepare_data
#
# with open(os.path.join("data/out", "train.pkl"), "rb") as f:
#     train = pickle.load(f)
#
# # trainer = pycrfsuite.Trainer(algorithm = 'ap',verbose=True)
# trainer = pycrfsuite.Trainer(algorithm = 'pa',verbose=True)
# #rainer = pycrfsuite.Trainer(algorithm = 'arow',verbose=True)
# trainer.set_params({
#                     'type':3,
#                      'c': 0.1, # coefficient for L1 penalty
# #                     'c2': 0.01,  # coefficient for L2 penalty
#                     'max_iterations': 2000,
#                     'feature.possible_transitions': False,
#                     'feature.possible_states': False
#                    })
#
# """
# Train five models
#
# """
# for i, data in enumerate(train):
#     temp = prepare_data(data)
#     for features, ylabel in temp:
#         trainer.append(features, ylabel)
#     trainer.train("exp_{}".format(i))
#     print("Model {} Trained".format(i))
#

# Train one model and see the result

# temp = prepare_data(train[4])
# for features, ylabel in temp:
#     trainer.append(features, ylabel)
# trainer.train("exp_{}".format(4))
# print("Model {} Trained".format(4))
#
"""
Prepare GT and Pred label sequence
"""
import os
import pickle
import pycrfsuite

from crfsuite_data import prepare_data
from reporting import StatsManager, pretty_print_report, pretty_rl_table

with open(os.path.join("data/out", "new_test.pkl"), "rb") as f:
    test = pickle.load(f)


data = test[0]

tagger = pycrfsuite.Tagger()
tagger.open('exp_{}'.format(5))


y_pred = []
y_true = []
feature_total = []

temp = prepare_data(data)
for features, ylabel in temp:

          feature_total.append(features)  #900*n

          y_true.append(ylabel)#900*n

          y_pred.append(tagger.tag(features))



"""
Prepare within_bbox feature values
"""

import numpy as np
f_num = []

for i, value in enumerate(feature_total): # feature total have 19 chunk of 900 frames each
    # print(i)
    # if i==1:
        # print(len(value)) # 900, value means one chunk of 900 frame feature and its result
    for j in value:
#         if j['is_card'] == True:
#             f_num.append(1)
        if j['is_face'] == True:
            f_num.append(2)
        elif j['is_dice'] == True:
            f_num.append(4)
        elif j['is_key'] == True:
            f_num.append(6)
        elif j['is_map'] == True:
            f_num.append(8)
        elif j['is_ball'] == True:
            f_num.append(10)
        else:
            f_num.append(np.nan)


"""
Print out evaluation result
"""

import numpy as np

from collections import defaultdict
from itertools import chain

from sklearn.metrics import classification_report, hamming_loss,accuracy_score

from sklearn.preprocessing import LabelBinarizer
from prettytable import PrettyTable

from constants import LABELS,INOUTLABELS,WITHOBJ


lb = LabelBinarizer()
y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))


tagset = set(lb.classes_)
tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
report = classification_report(y_true_combined,y_pred_combined,labels=[class_indices[cls] for cls in tagset],
                               target_names=tagset, output_dict=True)

# Onset_evaluation:
score = {"B-face":[],
            "B-dice":[],
            "B-key":[],
            "B-map":[],
            "B-ball":[],
            "B-card":[]
            }

for tag in lb.classes_:
    if tag.startswith("B-"):
        score = 0
        for i in range(len(y_true)):
            index_pred= get_beginning_index(y_pred[i],tag)
            index_gt = get_beginning_index(y_true[i],tag)
            score = score + onset_evaluation(index_gt, index_pred , 0.001)
            # if tag == "B-key":
            #     print(index_gt,index_pred)
            #     print(tag,score)

        score[tag].append(score)
    else:
        score[tag].append(0)

# print("Micro-average F1 score(same as overall accuracy): ", accuracy_score(y_true_combined, y_pred_combined))


def pretty_print_report(report,score):
    table = PrettyTable(["", "Precision", "Recall", "F1-Score", "Support","Onset_Delay_Per_Look"])
    for obj in report:
        if obj in WITHOBJ.keys():
            precision = report[obj]["precision"]
            recall = report[obj]["recall"]
            f1 = report[obj]["f1-score"]
            sup = report[obj]["support"]
            onset = score[obj]
            table.add_row([obj, "{:03.2f} ({:03.2f})".format(precision[0], precision[1]),
                           "{:03.2f} ({:03.2f})".format(recall[0], recall[1]),
                           "{:03.2f} ({:03.2f})".format(f1[0], f1[1]),
                           "{:03.2f} ({:03.2f})".format(sup[0], sup[1]),
                           "{:03.2f} ({:03.2f})".format(onset[0]/sup[0], onset[1]/sup[0])])

    print(table)


# pretty_print_report(report,score)




"""
Convert label sequence to number sequence
"""

from constants import INOUTLABELS,WITHOBJ
import numpy as np

gt_final = []
# print(y_true)
for i, value in enumerate (y_true):
    for key in value:
        if WITHOBJ[key] == 0:
            gt_final.append(0)
        else:
            gt_final.append(WITHOBJ[key])

pred_final = []

for i, value in enumerate (y_pred):
    for key in value:
        if WITHOBJ[key] == 0:
            pred_final.append(0)
        else:
            pred_final.append(WITHOBJ[key])




"""
Plot pred on GT, or plot feature on GT
"""

import matplotlib.pyplot as plt
plt.figure(figsize=[500,1])

# plt.plot(range(60,120),gt_final[60:120],'go',range(60,120),pred_final[60:120],'r^')
# plt.plot(range(len(gt_final)),gt_final,'go',range(len(pred_final)),pred_final,'ro')
# plt.plot(range(len(gt_final)),gt_final,'go',range(len(f_num)),f_num,'ro')
plt.plot(range(1000,2000),gt_final[1000:2000],'go',range(1000,2000),f_num[1000:2000],'ro')
#plt.plot(range(len(pred_final)),pred_final,'ro')
plt.ylabel('objects')
plt.xlabel('frame #')
plt.show()



"""
Print useful details of the model
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
#
# print("Top positive:")
# print_state_features(Counter(info.state_features).most_common(20))
#
# print("\nTop negative:")
# print_state_features(Counter(info.state_features).most_common()[-20:])

"""
Plot confusion matrix
"""
# import numpy as np
# import matplotlib.pyplot as plt
#
# from sklearn.metrics import confusion_matrix
#
# def plot_confusion_matrix(y_true, y_pred,
#                           normalize=False,
#                           title=None,
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if not title:
#         if normalize:
#             title = 'Normalized confusion matrix'
#         else:
#             title = 'Confusion matrix, without normalization'
#
#     # Compute confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     # Only use the labels that appear in the data
#     classes = ["O","B-face",
#     "I-face",
#     "B-dice",
#     "I-dice",
#     "B-key",
#     "I-key",
#     "B-map",
#     "I-map",
#     "B-ball",
#     "I-ball",
#     # 11:"B-card",
#     # 12:"I-card",
#     #
#     ]
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            # ... and label them with the respective list entries
#            xticklabels=classes, yticklabels=classes,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')
#
#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#
#     # Loop over data dimensions and create text annotations.
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     return ax
#
#
# np.set_printoptions(precision=2)
#
# # Plot non-normalized confusion matrix
# plot_confusion_matrix(gt_final, pred_final,
#                        title='Confusion matrix')
# plt.show()
