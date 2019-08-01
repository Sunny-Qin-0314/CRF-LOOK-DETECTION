import numpy as np
import argparse
import imutils
import time
import cv2
import os
import pandas as pd
import ast
import math
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
ap.add_argument("-b", "--bbox",
	help="path to bbox csv file")
ap.add_argument("-g", "--gaze",
	help="path to gaze csv file")
args = vars(ap.parse_args())

input_video_path= args["input"]
output_video_path = args["output"]
bbox_path= args["bbox"]
gaze_path = args["gaze"]
# print(args["bbox"])
# print(args["gaze"])

vs = cv2.VideoCapture(input_video_path)
writer = None


# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	total = -1


if bbox_path != None:
    with open(bbox_path) as f:
        object_data = f.readlines()
        # print(len(object_data))

if gaze_path != None:
    gaze_data = pd.read_csv(gaze_path)
    gaze_data = gaze_data[["world_timestamp", "world_index", "confidence", "norm_pos_x", "norm_pos_y"]]
    # print(len(gaze_data))


idx=0
# loop over frames from the video file stream
# while True:
#
# 	# read the next frame from the file
#     (grabbed, frame) = vs.read()
#
# 	# if the frame was not grabbed, then we have reached the end
# 	# of the stream
#     if not grabbed:
#         break
#
#     # start drawing
#     if gaze_path != None:
#     # if gaze_data != None :
#
#         # draw gaze dot to the frame
#         frame_data = gaze_data[gaze_data["world_index"] == idx]
#         filtered = frame_data[frame_data["confidence"] > 0.5]
#         x = filtered["norm_pos_x"].mean()
#         y = filtered["norm_pos_y"].mean()
#         if math.isnan(x) or math.isnan(y):
#             x, y = 0, 0
#         cv2.circle(frame,(int(x*1280),int((1-y)*720)), 4, (0,0,255), -1)
#
#     if bbox_path != None:
#         # print(len(object_data))
#         # draw bbox to the frame
#         if idx < len(object_data):
#             if object_data[idx] != "\n":
#                 a = ast.literal_eval(object_data[idx])
#                 if isinstance(a, tuple):
#                     for b in a:
#
#                         # print(object)
#                         obj = ast.literal_eval(b)
#
#                         startX = obj[2][0] - obj[2][2] / 2
#                         startY = obj[2][1] - obj[2][3] / 2
#                         endX = obj[2][0] + obj[2][2] / 2
#                         endY = obj[2][1] + obj[2][3] / 2
#                         cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), (0,255,0), 2)
#                         # cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
#
#     idx = idx+1
#
#     # check if the video writer is None
#     if writer is None:
# 		# initialize our video writer
#         fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#         writer = cv2.VideoWriter(output_video_path, fourcc, 30,(frame.shape[1], frame.shape[0]), True)
#
#     # write the output frame to disk
#     writer.write(frame)


# release the file pointers
# print("[INFO] cleaning up...")
# writer.release()
# vs.release()
