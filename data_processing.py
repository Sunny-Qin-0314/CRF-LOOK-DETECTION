import pandas as pd
import numpy as np
import os
import ast
import math
import pickle

from collections import defaultdict
from sklearn.model_selection import KFold

from constants import FILE_CHUNK, OBJECT_DETECTIONS, TEST_OBJECT_DETECTIONS ,TEST_GAZE_POSITIONS, TEST_GROUND_TRUTH, GAZE_POSITIONS, GROUND_TRUTH, NUMBER_OF_CV_FOLDS, WITHOBJ,INOUTLABELS,LABELS, \
    OBJECT_DET_DIR, GAZE_POS_DIR, GT_DIR


def bbox_normalized_coords(bbox):
    x = bbox[0] / 1280
    y = 1 - bbox[1] / 720
    w = bbox[2] / 1280
    h = bbox[3] / 720
    return x, y, w, h


def read_object_detections(filename):
    with open(os.path.join(OBJECT_DET_DIR, filename)) as f:
        object_data = f.readlines()

    frame_list = []
    bbox_list = []

    for line in object_data:
        frame = defaultdict(list)
        bbox = defaultdict(list)

        if line != "\n":
            a = ast.literal_eval(line)
            if isinstance(a, tuple):
                for b in a:
                    obj = ast.literal_eval(b)  #name,confidence,(x,y,w,h)
                    x, y, w, h = bbox_normalized_coords(obj[2])
                    obj_name = obj[0]
                    if obj_name == "cards":
                        obj_name = "card"

                    frame[obj_name].append((x, y, obj[1]))
                    bbox[obj_name + "_bbox"].append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
            else:
                obj = ast.literal_eval(a)
                x, y, w, h = bbox_normalized_coords(obj[2])
                obj_name = obj[0]
                if obj_name == "cards":
                    obj_name = "card"

                frame[obj_name].append((x, y, obj[1]))
                bbox[obj_name + "_bbox"].append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))

        frame_list.append(frame)
        bbox_list.append(bbox)

    return frame_list, bbox_list


# def read_gaze_file(filename):
#     gaze_data = pd.read_csv(os.path.join(GAZE_POS_DIR, filename))
#     gaze_data = gaze_data[["timestamp", "index", "confidence", "norm_pos_x", "norm_pos_y"]]
#     max_idx = int(gaze_data.iloc[-1]["index"] + 1)
#     tpf = (gaze_data.iloc[-1]["timestamp"] - gaze_data.iloc[0]["timestamp"]) / max_idx
#     gaze_list = []
#     displacement_list = []
#     velocity_list = []
#     x_temp = 0
#     y_temp = 0
#     for i in range(max_idx):
#         frame = gaze_data[gaze_data["index"] == i]
#         filtered = frame[frame["confidence"] > 0.5]
#         x = filtered["norm_pos_x"].mean()
#         y = filtered["norm_pos_y"].mean()
#         if math.isnan(x) or math.isnan(y):
#             x, y = gaze_list[-1]
#         gaze_list.append((x, y))

#     return gaze_list, max_idx, tpf

"""
not use tpf to calculate look frame number
instead use start_sec(in annotation file) compared with world_timestamp(in gaze file) to get world_index/frame number(in gaze file)

in read_gaze_file, output "world_start_time" and "world_index list"
in read_looks_gt_file, use the bisect algorithm to get the index of "world_index" list and then get the frame number
in main, change the output and input for read_gaze_file and read_looks_gt_file functions

"""
def read_gaze_file(filename):
    gaze_data = pd.read_csv(os.path.join(GAZE_POS_DIR, filename))
    try:
        gaze_data = gaze_data[["world_timestamp", "world_index", "confidence", "norm_pos_x", "norm_pos_y"]]
    #     max_idx = int(gaze_data.iloc[-1]["world_index"] + 1)
        max_idx = int(gaze_data.iloc[-1]["world_index"] )
        print(max_idx)
        world_start_time = gaze_data.iloc[0]["world_timestamp"]
        time_frame_data = gaze_data[["world_timestamp", "world_index"]]

        # tpf = (gaze_data.iloc[-1]["world_timestamp"] - gaze_data.iloc[0]["world_timestamp"]) / (max_idx+1)
        # tpf = (average_end_time - average_start_time) / (max_idx+1)
        # print(tpf)
        gaze_list = []
        displacement_list = []
        velocity_list = []
        x_temp = 0
        y_temp = 0
        for i in range(max_idx):
            frame = gaze_data[gaze_data["world_index"] == i]
            filtered = frame[frame["confidence"] > 0.5]
            x = filtered["norm_pos_x"].mean()
            y = filtered["norm_pos_y"].mean()
            if math.isnan(x) or math.isnan(y):
                if i == 0:
                    x, y = 0, 0
                else:
                    x, y = gaze_list[-1]
            gaze_list.append((x, y))
            dx = abs(x - x_temp)
            dy = abs(y - y_temp)
            d = (dx**2+dy**2)**0.5
            x_temp = x
            y_temp = y
            displacement_list.append(d)
    except KeyError:
        gaze_data = gaze_data[["timestamp", "index", "confidence", "norm_pos_x", "norm_pos_y"]]
        max_idx = int(gaze_data.iloc[-1]["index"] + 1)
        world_start_time = gaze_data.iloc[0]["timestamp"]
        time_frame_data = gaze_data[["timestamp", "ndex"]]

        gaze_list = []
        displacement_list = []
        velocity_list = []
        x_temp = 0
        y_temp = 0
        for i in range(max_idx):
            frame = gaze_data[gaze_data["index"] == i]
            filtered = frame[frame["confidence"] > 0.5]
            x = filtered["norm_pos_x"].mean()
            y = filtered["norm_pos_y"].mean()
            if math.isnan(x) or math.isnan(y):
                if i == 0:
                    x, y = 0, 0
                else:
                    x, y = gaze_list[-1]
            gaze_list.append((x, y))
            dx = abs(x - x_temp)
            dy = abs(y - y_temp)
            d = (dx**2+dy**2)**0.5
            x_temp = x
            y_temp = y
            displacement_list.append(d)

    return world_start_time,time_frame_data, gaze_list, max_idx, displacement_list



# def get_frame_gaze_dict(gaze_list, frame_list, bbox_list, max_idx):
#     out = {
#         "card": [],
#         "face": [],
#         "dice": [],
#         "key": [],
#         "map": [],
#         "ball": []
#     }
#     bbox_out = {"card_bbox": [],
#                 "face_bbox": [],
#                 "dice_bbox": [],
#                 "key_bbox": [],
#                 "map_bbox": [],
#                 "ball_bbox": []}

#     for i in range(max_idx):

#         pupil = gaze_list[i]
#         frame = frame_list[i]
#         bbox_l = bbox_list[i]

#         for key in out.keys():

#             if key in frame:
#                 dists = []

#                 for pt in frame[key]:
#                     dist = ((pupil[0] - pt[0]) ** 2 + (pupil[1] - pt[1]) ** 2) ** 0.5
#                     if math.isnan(dist):
#                         print(pupil, pt)
#                     else:
#                         dists.append(dist)

#                 min_dist = min(dists)
#                 out[key].append(min_dist)
#             else:
#                 out[key].append(1)

#         for key in bbox_out.keys():
#             is_in = False
#             if key in bbox_l:
#                 for bbox in bbox_l[key]:
#                     if ((bbox[0] <= pupil[0] <= bbox[2]) and
#                             (bbox[1] <= pupil[1] <= bbox[3])):
#                         is_in = True
#                         break
#             bbox_out[key].append(is_in)
#     out.update(bbox_out)
#     out["index"] = np.arange(max_idx)
#     return out

def get_frame_gaze_dict(gaze_list, frame_list, bbox_list, max_idx, displacement_list):
# def get_frame_gaze_dict(gaze_list, frame_list, bbox_list, max_idx):
    out = {
#        "card": [],
        "face": [],
        "dice": [],
        "key": [],
        "map": [],
        "ball": []
    }
    bbox_out = {
#                "card_bbox": [],
                "face_bbox": [],
                "dice_bbox": [],
                "key_bbox": [],
                "map_bbox": [],
                "ball_bbox": []}

    displacement_out = {"gaze_displacement": []
                       }

    for i in range(max_idx):
        pupil = gaze_list[i]
        frame = frame_list[i]
        bbox_l = bbox_list[i]
        displacement = displacement_list[i]

        for key in out.keys():

            if key in frame:
                dists = []
                for pt in frame[key]:
                    dist = ((pupil[0] - pt[0]) ** 2 + (pupil[1] - pt[1]) ** 2) ** 0.5
                    if math.isnan(dist):
                        print(pupil, pt)
                    else:
                        dists.append(dist)

                min_dist = min(dists)
                out[key].append(min_dist)
            else:
                out[key].append(1)

        for key in bbox_out.keys():
            is_in = False
            if key in bbox_l:
                for bbox in bbox_l[key]:
                    if ((bbox[0] <= pupil[0] <= bbox[2]) and
                            (bbox[1] <= pupil[1] <= bbox[3])):
                        is_in = True
                        break
            bbox_out[key].append(is_in)

        for key in displacement_out.keys():

                    displacement_out[key].append(displacement)
        # if(i==12040):
        #     print(bbox_out["key_bbox"][i])


    out.update(bbox_out)
    # print(out["key_bbox"][12040])
    out.update(displacement_out)
    out["index"] = np.arange(max_idx)
    return out

from bisect import bisect_left

def takeClosest(myList, myNumber):
    pos = bisect_left(myList, myNumber)
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber >= myNumber - before:
       return pos-1
    else:
       return pos

def read_looks_gt_file(filename, out, max_idx, time_frame_data, world_start_time):
    looks = pd.read_csv(os.path.join(GT_DIR, filename))[["object", "start_sec", "end_sec"]]
    looks = looks.sort_values("start_sec")
    out["look"] = np.array([0] * max_idx)

    for row in looks.values:
        world_time_look_start = row[1] + world_start_time
        world_time_look_end = row[2] + world_start_time
        pos_start = takeClosest(time_frame_data["world_timestamp"],world_time_look_start)
        pos_end = takeClosest(time_frame_data["world_timestamp"],world_time_look_end)

        start =  int(time_frame_data.iloc[pos_start]["world_index"])
        end = int(time_frame_data.iloc[pos_end]["world_index"]+1)
        out["look"][start]= WITHOBJ["B-"+row[0]]
        out["look"][start+1:end] = WITHOBJ["I-"+row[0]]
        # out["look"][start:end] = LABELS[row[0]] # return object number

    out["look"] = list(out["look"])
    # print(len(out["look"]))
    # print(out["look"][9820:9900])
    # print(out["key_bbox"][12040])
    return out


def create_chunks(final_dicts, chunk_size):
    chunks = []
    for i, data in enumerate(final_dicts):
        num_frames = len(data["look"])
        num_chunks = num_frames // chunk_size
        print(i, num_chunks)
        for j in range(num_chunks):
            chunk = {}
            # print(data.keys())
            for key in data.keys():
                chunk[key] = data[key][j*chunk_size:(j+1)*chunk_size]
            chunks.append(chunk)
    return chunks


def main():


# Prepare training set and validation set


    obj_data = []
    bbox_data = []

    for file in OBJECT_DETECTIONS:
        frame_list, bbox_list = read_object_detections(file)
        obj_data.append(frame_list)
        bbox_data.append(bbox_list)

    gaze_data = []
    max_idxs = []
    tpfs = []
    world_start_times = []
    time_frame_datas = []
    displacement_data = []

#     for file in GAZE_POSITIONS:
#         gz, m_idx, tpf = read_gaze_file(file)
#         gaze_data.append(gz)
#         max_idxs.append(m_idx)
#         tpfs.append(tpf)

    for file in GAZE_POSITIONS:
        world_start_time, time_frame_data, gz, m_idx, ds = read_gaze_file(file)
        gaze_data.append(gz)
        max_idxs.append(m_idx)
        world_start_times.append(world_start_time)
        time_frame_datas.append(time_frame_data)
        displacement_data.append(ds)

    out_dicts = []
    # for obj, bbx, gz, midx in zip(obj_data, bbox_data, gaze_data, max_idxs):
        # out_dicts.append(get_frame_gaze_dict(gz, obj, bbx, midx))
    for obj, bbx, gz, midx, ds in zip(obj_data, bbox_data, gaze_data, max_idxs, displacement_data):
        out_dicts.append(get_frame_gaze_dict(gz, obj, bbx, midx, ds))
    # print(out_dicts[0]["key_bbox"][12040])
    final_dicts = []
    for file, out, midx, world_start_time, time_frame_data in zip(GROUND_TRUTH, out_dicts, max_idxs, world_start_times, time_frame_datas):
        final_dicts.append(read_looks_gt_file(file, out, midx, time_frame_data, world_start_time))



# Prepare test set

    test_obj_data = []
    test_bbox_data = []

    for file in TEST_OBJECT_DETECTIONS:
        frame_list, bbox_list = read_object_detections(file)
        test_obj_data.append(frame_list)
        test_bbox_data.append(bbox_list)

    test_gaze_data = []
    test_max_idxs = []
    test_world_start_times = []
    test_time_frame_datas = []
    test_displacement_data = []

    for file in TEST_GAZE_POSITIONS:
        world_start_time, time_frame_data, gz, m_idx, ds = read_gaze_file(file)
        test_gaze_data.append(gz)
        test_max_idxs.append(m_idx)
        test_world_start_times.append(world_start_time)
        test_time_frame_datas.append(time_frame_data)
        test_displacement_data.append(ds)

    test_out_dicts = []
    # for obj, bbx, gz, midx in zip(obj_data, bbox_data, gaze_data, max_idxs):
    # out_dicts.append(get_frame_gaze_dict(gz, obj, bbx, midx))
    for obj, bbx, gz, midx, ds in zip(test_obj_data, test_bbox_data, test_gaze_data, test_max_idxs, test_displacement_data):
        out_dicts.append(get_frame_gaze_dict(gz, obj, bbx, midx, ds))

    test_final_dicts = []
    for file, out, midx, world_start_time, time_frame_data in zip(TEST_GROUND_TRUTH, test_out_dicts, test_max_idxs, test_world_start_times, test_time_frame_datas):
        test_final_dicts.append(read_looks_gt_file(file, out, midx, time_frame_data, world_start_time))




# Generate those sets and save those sets into folder 

    out_dir = os.path.join("data", "out")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # print(len(final_dicts))

    chunk_dicts = np.array(create_chunks(final_dicts, FILE_CHUNK))
    # print(chunk_dicts[13]["key_bbox"][340:350])
    with open(os.path.join(out_dir, "data.pkl"), "wb") as f:
        pickle.dump(chunk_dicts, f)

    train = []
    validation = []
    test = []

    kf = KFold(n_splits=NUMBER_OF_CV_FOLDS)
            # print(kf)
    for train_split, validation_split in kf.split(chunk_dicts):
                # print(test_split)
        train.append(chunk_dicts[train_split])

        validation.append(chunk_dicts[validation_split])
                # print(chunk_dicts[[0,1,2,3]])

    test.append(test_final_dicts)


    with open(os.path.join(out_dir, "train.pkl"), "wb") as f:
        pickle.dump(train, f)

    with open(os.path.join(out_dir, "validation.pkl"), "wb") as f:
        pickle.dump(validation, f)

    with open(os.path.join(out_dir, "test.pkl"), "wb") as f:
        pickle.dump(test, f)

if __name__ == "__main__":
    main()
