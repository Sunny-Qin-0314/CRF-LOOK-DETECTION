import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from constants import REV_LABELS, REV_INOUTLABELS, REV_WITHOBJ


def get_crf_features(out):
    df = pd.DataFrame(out)
    features = []
    ylabel = []
    y_num_label = []
    for i in range(0, len(df)):
        frame = df.iloc[i]

        f = {
             # "bias":0.1,
#              "card_distance":frame["card"],
             "dice_distance":frame["dice"],
             "map_distance":frame["map"],
             "key_distance":frame["key"],
             "face_distance":frame["face"],
             "ball_distance":frame["ball"],
             #
#             "is_card":frame["card_bbox"],
             "is_face":frame["face_bbox"],
             "is_dice":frame["dice_bbox"],
             "is_key":frame["key_bbox"],
             "is_map":frame["map_bbox"],
             "is_ball":frame["ball_bbox"],
             #
             "gaze_displacement":frame["gaze_displacement"],
             # "gaze_confidence" : frame["gaze_confidence"]
            }

        if i > 0:
            last_frame = df.iloc[i-1]
            f.update(

            {
            # "prev_is_card":last_frame["card_bbox"],
             "prev_is_dice":last_frame["dice_bbox"],
             "prev_is_map":last_frame["map_bbox"],
             "prev_is_key":last_frame["key_bbox"],
             "prev_is_face":last_frame["face_bbox"],
             "prev_is_ball":last_frame["ball_bbox"],
#              "prev_card_disctance":last_frame["card"],
             "prev_dice_disctance":last_frame["dice"],
             "prev_map_distance":last_frame["map"],
             "prev_key_distance":last_frame["key"],
             "prev_face_distance":last_frame["face"],
             "prev_ball_distance":last_frame["ball"],

             "prev_gaze_displacement":last_frame["gaze_displacement"],
             # "prev_gaze_confidence":last_frame["gaze_confidence"]
             }
             )

        if i < len(df)-1:
            next_frame = df.iloc[i+1]
            f.update(

            {
            # "next_is_card":next_frame["card_bbox"],
             "next_is_dice":next_frame["dice_bbox"],
             "next_is_map":next_frame["map_bbox"],
             "next_is_key":next_frame["key_bbox"],
             "next_is_face":next_frame["face_bbox"],
             "next_is_ball":next_frame["ball_bbox"],
#              "next_card_distance":next_frame["card"],
             "next_dice_distance":next_frame["dice"],
             "next_map_distance":next_frame["map"],
             "next_key_distance":next_frame["key"],
             "next_face_distance":next_frame["face"],
             "next_ball_distance":next_frame["ball"],
             #
             "next_gaze_displacement":next_frame["gaze_displacement"],
             # "next_gaze_confidence":next_frame["gaze_confidence"]
                 }

             )
        features.append(f)
        ylabel.append(REV_WITHOBJ[frame["look"]])
        y_num_label.append(frame["look"])
    return features, ylabel, y_num_label


def get_label(arr):
    values, counts = np.unique(arr, return_counts=True)
    idx = np.argmax(counts)
    return values[idx]


def prepare_data(data_dicts):
    features_list = []
    ylabel_list = []
    num_label = []
    # print(data_dicts[13]["key_bbox"][340:350])
    for fdict in data_dicts:
        features, ylabel, y_num_label = get_crf_features(fdict)
        features_list.append(features)
        ylabel_list.append(ylabel)
        num_label.append(y_num_label)

        # print(ylabel_list[0][200:300])
    data = zip(features_list, ylabel_list)

    return data
