import numpy as np

from collections import defaultdict
from itertools import chain

from sklearn.metrics import classification_report , accuracy_score
from sklearn.preprocessing import LabelBinarizer
from prettytable import PrettyTable

from constants import LABELS,WITHOBJ


class StatsManager:

    def __init__(self, supp_thres=0):
        self.reports = []
        self.y_pred = []
        self.y_true = []
        self.hammingloss =[]
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
        print("Micro-average F1 score(same as overall accuracy): ", accuracy_score(y_true_combined, y_pred_combined))

        self.reports.append(report)

    def summarize(self):
        summary = defaultdict(lambda: defaultdict(list))

        for report in self.reports:
            # print(report.keys())
            for key in report.keys():
                # print(key)
                for metric in report[key].keys():
                    # if report[key]["support"] > self.support_threshold:
                        summary[key][metric].append(report[key][metric])

        report = defaultdict(dict)
        temp2 = []
        # print(summary.keys())
        for key in summary.keys():
            metrics = defaultdict()
            temp2 = []
            for metric in summary[key].keys():
                temp = summary[key][metric]
                for i in temp:
                    if (i != 0):
                        temp2.append(i)
                metrics[metric] = [np.mean(temp2), np.std(temp2)]
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
    table = PrettyTable(["", "Precision", "Recall", "F1-Score", "Support"])
    for obj in report:
        # print(obj)
        if obj in WITHOBJ.keys():
            precision = report[obj]["precision"]
            recall = report[obj]["recall"]
            f1 = report[obj]["f1-score"]
            sup = report[obj]["support"]
            table.add_row([obj, "{:03.2f} ({:03.2f})".format(precision[0], precision[1]),
                           "{:03.2f} ({:03.2f})".format(recall[0], recall[1]),
                           "{:03.2f} ({:03.2f})".format(f1[0], f1[1]),
                           "{:03.2f} ({:03.2f})".format(sup[0], sup[1])])

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
