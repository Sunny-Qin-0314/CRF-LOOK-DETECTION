import pandas as pd
import numpy as np
import os
import ast
import math
import pickle

from bisect import bisect_left

def take_closest(myList, myNumber):
    """
    # Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before

"""
onset_evaluation = look_diff + w * onset_diff

The inputs are the list of frame numbers that B-obj tag showing in the GT and Pred sequence
[49,530] means that the 49th and 530th frames are the B-obj tag

If the length are same, we do subtraction one by one, else we take the closest number in the list to subtract.
"""
def onset_evaluation(index_gt, index_pred, weight):
    onset = 0
    if len(index_gt) == len(index_pred):
        for i in range(len(index_gt)):
            onset = onset + abs(index_pred[i] - index_gt[i])

    elif len(index_gt) < len(index_pred):
        for i in range(len(index_gt)):
            pred_closest = take_closest(index_pred,index_gt[i])
            onset = onset + abs(pred_closest - index_gt[i])

    else:
        for i in range(len(index_pred)):
            gt_closest = take_closest(index_gt,index_pred[i])
            onset = onset + abs(gt_closest - index_pred[i])

    evaluation_result = abs(len(index_gt) - len(index_pred)) + weight * onset

    return evaluation_result

def get_beginning_index(num_sequence, sub_class):
    return [index for index, value in enumerate(num_sequence) if value == sub_class]
