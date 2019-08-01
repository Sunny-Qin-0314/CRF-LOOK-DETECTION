import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gaze_data = pd.read_csv("./data/gaze/2019_06_13_000_gaze_positions.csv")
gaze_data = gaze_data[["world_timestamp", "world_index", "confidence", "norm_pos_x", "norm_pos_y"]]
max_idx = int(gaze_data.iloc[-1]["world_index"] )
print(max_idx)

for i in range(max_idx):
    frame = gaze_data[gaze_data["world_index"] == i]
    filtered = frame[frame["confidence"] > 0.5]
    x = filtered["norm_pos_x"].mean()
    y = filtered["norm_pos_y"].mean()
    # print(x,y)


    filename = "{:06d}.jpg".format(i)
    # print("{:06d}".format(i))
    im = plt.imread("./data/2019_06_13_000_test/{}".format(filename))
    implot = plt.imshow(im)


    plt.plot(x*1280,(1-y)*720,'ro',markersize=5)


    plt.savefig('./data/06_13_000_gaze_with_bbox/{}'.format(filename))
    plt.clf()
    # plt.show()
