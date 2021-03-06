import numpy as np

from collections import defaultdict
from itertools import chain

from sklearn.metrics import classification_report , accuracy_score
from sklearn.preprocessing import LabelBinarizer
from prettytable import PrettyTable

from constants import LABELS,WITHOBJ
from onset_evaluation import *

def convertB_to_I(sequence):
    new = []
    for i in range(len(sequence)):
        if sequence[i].startswith("B-"):
            new.append(sequence[i].replace("B-","I-"))
        else:
            new.append(sequence[i])
    return new

def accuracy(y_true, y_pred):
    TN = 0
    TP = 0
    total = 0
    for i in range(len(y_true)):
        new_y_true = convertB_to_I(y_true[i])
        new_y_pred = convertB_to_I(y_pred[i])
        total = total + len(new_y_true)
        for j in range(len(new_y_true)):
            if new_y_true[j] == new_y_pred[j] and new_y_true[j]=="O":
                TN = TN + 1
            elif new_y_true[j] == new_y_pred[j] :
                TP = TP +1
    print("TN:{}, TP:{}, total(include TN):{}".format(TN,TP,total))
    accuracy = TP / (total - TN)
    return accuracy

class StatsManager:

    def __init__(self, supp_thres=0):
        self.reports = []
        self.y_pred = []
        self.y_true = []
        self.score = {"B-face":[],
                    "B-dice":[],
                    "B-key":[],
                    "B-map":[],
                    "B-ball":[],
                    "B-card":[]
                    }
        self.support_threshold = supp_thres

    def transform(self, data):
        t_data = []
        for l in data:
            ele = []
            for e in l:
                ele.append(WITHOBJ[e])
            t_data.append(ele)
        return t_data

    def append_report(self, y_true, y_pred):
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

        # Precision, recall, f1, support:

        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_)
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        report = classification_report(
            y_true_combined,
            y_pred_combined,
            labels=[class_indices[cls] for cls in tagset],
            target_names=tagset,
            output_dict=True
        )

        # Onset_evaluation:

        for tag in lb.classes_:
            if tag.startswith("B-"):
                score = 0
                for i in range(len(y_true)):
                    index_pred= get_beginning_index(y_pred[i],tag)
                    index_gt = get_beginning_index(y_true[i],tag)
                    score = score + onset_evaluation(index_gt, index_pred , 1)
                    # if tag == "B-key":
                    #     print(index_gt,index_pred)
                    #     print(tag,score)

                self.score[tag].append(score)

        # Accuracy evaluation:
        new_accuracy_measure = accuracy(y_true, y_pred)
        print("New_accuracy: {:03.2f}".format(new_accuracy_measure))
        print("Micro-average F1 score(same as overall accuracy): {:03.2f}".format(accuracy_score(y_true_combined, y_pred_combined)))

        self.reports.append(report)


    def summarize(self):
        """
        I change the mean, std calculation here. The reason is written as the comment in line 105
        """
        summary = defaultdict(lambda: defaultdict(list))

        for report in self.reports:
            for key in report.keys():
                for metric in report[key].keys():
                    if report[key]["support"] > self.support_threshold:
                        summary[key][metric].append(report[key][metric])


        report = defaultdict(dict)
        # temp2 = []
        for key in summary.keys():
            metrics = defaultdict()
            # temp2 = []
            for metric in summary[key].keys():
                temp = summary[key][metric]
                # for i in temp:
                #     if (i != 0):
                #         temp2.append(i)  # temp2 contains all the different matrix values, that doesn't make sense. temp2 never be empty in this for-loop
                metrics[metric] = [np.mean(temp), np.std(temp)]

            if key.startswith("B-"):
                metrics["Onset_Delay"] = [np.mean(self.score[key]), np.std(self.score[key])]
            else:
                metrics["Onset_Delay"] = [0,0]

            report[key] = metrics
        return report, summary

    # def runlength_report(self):
    #     report = defaultdict(list)
    #     objects = ["card", "face", "dice", "key", "map", "ball"]
    #
    #     for pred_list, truth_list in zip(self.y_pred, self.y_true):
    #         counter = defaultdict(lambda :[0]*2)
    #         for pred, truth in zip(pred_list, truth_list):
    #             pred = np.array(pred)
    #             truth = np.array(truth)
    #             for obj in objects:
    #                 p = (pred == obj)
    #                 t = (truth == obj)
    #                 tp, fp = runlength(p, t, 4, 3)
    #                 counter[obj][0] += tp
    #                 counter[obj][1] += fp
    #         for obj in WITHOBJ.keys():
    #             if sum(counter[obj]) > 0:
    #                 report[obj].append(counter[obj][0] / sum(counter[obj]))
    #
    #     return report


def pretty_print_report(report):
    table = PrettyTable(["", "Precision", "Recall", "F1-Score", "Support","Onset_Delay_Per_Look"])
    for obj in report:
        if obj in WITHOBJ.keys():
            precision = report[obj]["precision"]
            recall = report[obj]["recall"]
            f1 = report[obj]["f1-score"]
            sup = report[obj]["support"]
            onset = report[obj]["Onset_Delay"]
            table.add_row([obj, "{:03.2f} ({:03.2f})".format(precision[0], precision[1]),
                           "{:03.2f} ({:03.2f})".format(recall[0], recall[1]),
                           "{:03.2f} ({:03.2f})".format(f1[0], f1[1]),
                           "{:03.2f} ({:03.2f})".format(sup[0], sup[1]),
                           "{:03.2f} ({:03.2f})".format(onset[0]/sup[0], onset[1]/sup[0])])

    print(table)


# def runlength(pred, gt, window, num_ones):
#     tp = 0
#     fp = 0
#     if np.sum(pred) > 3:
#         for i in range(len(pred)-window):
#             if np.sum(pred[i:i+window]) >= num_ones:
#                 if np.sum(gt[i:i+window]) >= num_ones:
#                     tp += 1
#                 else:
#                     fp += 1
#                 break

#     return tp, fp


def pretty_rl_table(report):
    table = PrettyTable(["", "Precision"])
    for obj in sorted(report.keys()):
        m = np.mean(report[obj])
        s = np.std(report[obj])

        table.add_row([obj, "{:03.2f} ({:03.2f})".format(m, s)])
    print(table)
