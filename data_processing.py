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

"""
Get diagonal length of bbox
"""
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
            # print(a)
            if isinstance(a, tuple):
                for b in a:
                    obj = ast.literal_eval(b)  #name,confidence,(x,y,w,h)
                    x, y, w, h = bbox_normalized_coords(obj[2])
                    obj_name = obj[0]
                    if obj_name == "cards":
                        obj_name = "card"
                    dist = ((w/2)**2+ (h/2)**2)**0.5
                    frame[obj_name].append((x, y, obj[1],dist))
                    bbox[obj_name + "_bbox"].append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
            else:
                obj = ast.literal_eval(a)
                x, y, w, h = bbox_normalized_coords(obj[2])
                obj_name = obj[0]
                if obj_name == "cards":
                    obj_name = "card"
                dist = ((w/2)**2+ (h/2)**2)**0.5
                frame[obj_name].append((x, y, obj[1],dist))
                # frame[obj_name].append((x, y, obj[1]))
                bbox[obj_name + "_bbox"].append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))

        frame_list.append(frame)
        bbox_list.append(bbox)

    return frame_list, bbox_list



"""
Not use tpf to calculate look frame number
instead use start_sec(in annotation file) compared with world_timestamp(in gaze file) to get frame number(in gaze file)

Main changes:
in read_gaze_file, output "world_start_time" and "world_index list"
in read_looks_gt_file, use the bisect algorithm to get the index of "world_index" list and then get the frame number
in main, change the output and input for read_gaze_file and read_looks_gt_file functions

Get displacement from the previous frame and current frame
"""
def read_gaze_file(filename):
    gaze_data = pd.read_csv(os.path.join(GAZE_POS_DIR, filename))
    try:  # Use for new glasses
        gaze_data = gaze_data[["world_timestamp", "world_index", "confidence", "norm_pos_x", "norm_pos_y"]]
        max_idx = int(gaze_data.iloc[-1]["world_index"] + 1)
        print(max_idx)
        world_start_time = gaze_data.iloc[0]["world_timestamp"]
        time_frame_data = gaze_data[["world_timestamp", "world_index"]]

        gaze_list = []
        displacement_list = []
        confidence_list = []
        velocity_list = []
        x_temp = 0
        y_temp = 0
        for i in range(max_idx):
            frame = gaze_data[gaze_data["world_index"] == i]
            filtered = frame[frame["confidence"] > 0.5]
            confidence = filtered["confidence"].mean()
            x = filtered["norm_pos_x"].mean()
            y = filtered["norm_pos_y"].mean()
            if math.isnan(x) or math.isnan(y):
                if i == 0:
                    x, y = 0, 0
                else:
                    x, y = gaze_list[-1]
            gaze_list.append((x, y))
            confidence_list.append(confidence)

            dx = abs(x - x_temp)
            dy = abs(y - y_temp)
            d = (dx**2+dy**2)**0.5
            x_temp = x
            y_temp = y
            displacement_list.append(d)

    except KeyError: # Use for old glasses
        gaze_data = gaze_data[["timestamp", "index", "confidence", "norm_pos_x", "norm_pos_y"]]
        max_idx = int(gaze_data.iloc[-1]["index"] + 1)
        world_start_time = gaze_data.iloc[0]["timestamp"]
        time_frame_data = gaze_data[["timestamp", "ndex"]]

        gaze_list = []
        displacement_list = []
        confidence_list = []
        velocity_list = []
        x_temp = 0
        y_temp = 0
        for i in range(max_idx):
            frame = gaze_data[gaze_data["index"] == i]
            filtered = frame[frame["confidence"] > 0.5]
            confidence = filtered["confidence"].mean()
            x = filtered["norm_pos_x"].mean()
            y = filtered["norm_pos_y"].mean()
            if math.isnan(x) or math.isnan(y):
                if i == 0:
                    x, y = 0, 0
                else:
                    x, y = gaze_list[-1]
            gaze_list.append((x, y))
            confidence_list.append(confidence)

            dx = abs(x - x_temp)
            dy = abs(y - y_temp)
            d = (dx**2+dy**2)**0.5
            x_temp = x
            y_temp = y
            displacement_list.append(d)

    return world_start_time,time_frame_data, gaze_list, max_idx, displacement_list,confidence_list


"""
Use relative distance to the bbox as the feature
dist = norm_dist(gaze,center_bbox) / norm_dist(0.5* diagonal_bbox)

Add one more feature: displacement between the current frame and previous frame

Gaze_confidence feature does not have a good performance on the testset, so maybe not use it anymore
"""

def get_frame_gaze_dict(gaze_list, frame_list, bbox_list, max_idx, displacement_list, confidence_list):
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

    confidence_out = {"gaze_confidence": []}

    for i in range(max_idx):
        pupil = gaze_list[i]
        frame = frame_list[i]
        bbox_l = bbox_list[i]
        displacement = displacement_list[i]
        confidence = confidence_list[i]

        for key in out.keys():

            if key in frame:
                dists = []
                for pt in frame[key]:
                    dist = ((pupil[0] - pt[0]) ** 2 + (pupil[1] - pt[1]) ** 2) ** 0.5
                    norm_dist = dist / pt[3]
                    if math.isnan(norm_dist):
                        print(pupil, pt)
                    else:
                        dists.append(norm_dist)

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

        displacement_out["gaze_displacement"].append(displacement)
        confidence_out["gaze_confidence"].append(confidence)

    out.update(bbox_out)
    out.update(displacement_out)
    out.update(confidence_out)
    out["index"] = np.arange(max_idx)
    return out

from bisect import bisect_left

def takeClosest(myList, myNumber):
    pos = bisect_left(myList, myNumber)
    if pos == 0 :
        return 0
    if pos == len(myList):
        return pos
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber >= myNumber - before:
       return pos-1
    else:
       return pos

"""
Seperate B-obj tag and I-obj tag
"""
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
    return out


def create_chunks(final_dicts, chunk_size):
    chunks = []
    for i, data in enumerate(final_dicts):
        num_frames = len(data["look"])
        num_chunks = num_frames // chunk_size
        print(i, num_chunks)
        for j in range(num_chunks):
            chunk = {}
            for key in data.keys():
                chunk[key] = data[key][j*chunk_size:(j+1)*chunk_size]
            chunks.append(chunk)
    return chunks


def main():

# Prepare feature data

    obj_data = []
    bbox_data = []

    for file in OBJECT_DETECTIONS:
        print(file)
        frame_list, bbox_list = read_object_detections(file)
        obj_data.append(frame_list)
        bbox_data.append(bbox_list)

    gaze_data = []
    max_idxs = []
    tpfs = []
    world_start_times = []
    time_frame_datas = []
    displacement_data = []
    confidence_data = []

    for file in GAZE_POSITIONS:
        world_start_time, time_frame_data, gz, m_idx, ds, confidence = read_gaze_file(file)
        gaze_data.append(gz)
        max_idxs.append(m_idx)
        world_start_times.append(world_start_time)
        time_frame_datas.append(time_frame_data)
        displacement_data.append(ds)
        confidence_data.append(confidence)

    out_dicts = []
    for obj, bbx, gz, midx, ds, con in zip(obj_data, bbox_data, gaze_data, max_idxs, displacement_data,confidence_data):
        out_dicts.append(get_frame_gaze_dict(gz, obj, bbx, midx, ds, con))


    final_dicts = []
    for file, out, midx, world_start_time, time_frame_data in zip(GROUND_TRUTH, out_dicts, max_idxs, world_start_times, time_frame_datas):
        final_dicts.append(read_looks_gt_file(file, out, midx, time_frame_data, world_start_time))

    print(len(final_dicts))

# Prepare test data

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
    test_confidence_data = []

    for file in TEST_GAZE_POSITIONS:
        world_start_time, time_frame_data, gz, m_idx, ds ,con = read_gaze_file(file)
        test_gaze_data.append(gz)
        test_max_idxs.append(m_idx)
        test_world_start_times.append(world_start_time)
        test_time_frame_datas.append(time_frame_data)
        test_displacement_data.append(ds)
        test_confidence_data.append(con)

    test_out_dicts = []
    for obj, bbx, gz, midx, ds, con in zip(test_obj_data, test_bbox_data, test_gaze_data, test_max_idxs, test_displacement_data, test_confidence_data):
        test_out_dicts.append(get_frame_gaze_dict(gz, obj, bbx, midx, ds, con))

    test_final_dicts = []
    for file, out, midx, world_start_time, time_frame_data in zip(TEST_GROUND_TRUTH, test_out_dicts, test_max_idxs, test_world_start_times, test_time_frame_datas):
        test_final_dicts.append(read_looks_gt_file(file, out, midx, time_frame_data, world_start_time))



"""
Generate those sets and save those sets into folder
"""
    # Video level cross validation (new_train, new_test) : every video to be the test set, others to be the train set

    new_train =[]
    new_test = []

    for i in range(len(final_dicts)):
        new_test.append(np.array([final_dicts[i]]))
        new_train.append(np.array(final_dicts[0:i]+final_dicts[i+1:len(final_dicts)]))


    out_dir = os.path.join("data", "out")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Chunk level cross validation (train, validation, test) several chunks to be the test set, other chunks to be the train set

    chunk_dicts = np.array(create_chunks(final_dicts, FILE_CHUNK))

    with open(os.path.join(out_dir, "data.pkl"), "wb") as f:
        pickle.dump(chunk_dicts, f)

    train = []
    validation = []
    test = []


    kf = KFold(n_splits=NUMBER_OF_CV_FOLDS)
    for train_split, validation_split in kf.split(chunk_dicts):
        print(train_split, validation_split)
        train.append(chunk_dicts[train_split])
        validation.append(chunk_dicts[validation_split])

    test.append(test_final_dicts)


    with open(os.path.join(out_dir, "train.pkl"), "wb") as f:
        pickle.dump(train, f)

    with open(os.path.join(out_dir, "validation.pkl"), "wb") as f:
        pickle.dump(validation, f)

    with open(os.path.join(out_dir, "test.pkl"), "wb") as f:
        pickle.dump(test, f)

    with open(os.path.join(out_dir, "new_train.pkl"), "wb") as f:
        pickle.dump(new_train, f)

    with open(os.path.join(out_dir, "new_test.pkl"), "wb") as f:
        pickle.dump(new_test, f)


if __name__ == "__main__":
    main()
