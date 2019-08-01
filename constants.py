import os
import numpy as np

# constants to be used for the project
ROOT = "data/"
OBJECT_DET_DIR = os.path.join(ROOT, "object_detections")
GAZE_POS_DIR = os.path.join(ROOT, "gaze")
GT_DIR = os.path.join(ROOT, "annotations")


#6_19_002 6_26_000 6_26_001 have offset problem

# OBJECT_DETECTIONS = ["Andy.csv", "Daniel.csv", "2018_07_17_001.csv",
#                      "2018_07_24_003.csv", "2018_07_17_004.csv"]  # "2018_07_25_000.csv"]
# GAZE_POSITIONS = ["Andy_gaze_positions.csv", "Daniel_gaze_positions.csv",
#                   "2018_07_17_001_gaze_positions.csv", "2018_07_24_003_gaze_positions.csv",
#                   "2018-07-17-004_gaze_positions.csv"]  # "2018_07_25_gaze_positions.csv"]
# GROUND_TRUTH = ["Andy_annotated.csv", "daniel_annotated.csv", "7_17_001_annotated.csv",
#                 "7_24_003_annotated.csv", "2018-07-17-004_annotated.csv"]  # "7_25_000_annotated.csv"]


# OBJECT_DETECTIONS = ["2019_06_13_000.csv", "2019_06_13_002.csv", "2019_06_19_000.csv", "2019_06_19_001.csv", "2019_06_19_002.csv",
#                      "2019_06_26_000.csv", "2019_06_26_001.csv", "2019_06_26_002.csv","Andy.csv", "Daniel.csv", "2018_07_17_001.csv",
#                       "2018_07_24_003.csv", "2018_07_17_004.csv"]

# GAZE_POSITIONS = ["2019_06_13_000_gaze_positions.csv", "2019_06_13_002_gaze_positions.csv", "2019_06_19_000_gaze_positions.csv",                                 "2019_06_19_001_gaze_positions.csv", "2019_06_19_002_gaze_positions.csv","2019_06_26_000_gaze_positions.csv",                                 "2019_06_26_001_gaze_positions.csv", "2019_06_26_002_gaze_positions.csv","Andy_gaze_positions.csv",                                           "Daniel_gaze_positions.csv", "2018_07_17_001_gaze_positions.csv", "2018_07_24_003_gaze_positions.csv",                                         "2018-07-17-004_gaze_positions.csv", "2018_07_25_gaze_positions.csv"]

# GROUND_TRUTH = ["2019_06_13_000_annotated.csv", "2019_06_13_002_annotated.csv", "2019_06_19_000_annotated.csv",                                               "2019_06_19_001_annotated.csv", "2019_06_19_002_annotated.csv", "2019_06_26_000_annotated.csv",                                               "2019_06_26_001_annotated.csv", "2019_06_26_002_annotated.csv","Andy_annotated.csv", "daniel_annotated.csv",
#                 "7_17_001_annotated.csv","7_24_003_annotated.csv", "2018-07-17-004_annotated.csv"]


# OBJECT_DETECTIONS = ["2019_06_13_000.csv", "2019_06_13_002.csv", "2019_06_19_000.csv", "2019_06_19_001.csv",
#                        "2019_06_26_002.csv"]
OBJECT_DETECTIONS = ["2019_06_26_002.csv","2019_06_13_002.csv"]
# GAZE_POSITIONS = ["2019_06_13_000_gaze_positions.csv", "2019_06_13_002_gaze_positions.csv", "2019_06_19_000_gaze_positions.csv",                                 "2019_06_19_001_gaze_positions.csv", "2019_06_26_002_gaze_positions.csv"]
GAZE_POSITIONS = ["2019_06_26_002_gaze_positions.csv","2019_06_13_002_gaze_positions.csv"]

# GROUND_TRUTH = ["2019_06_13_000_annotated.csv", "2019_06_13_002_annotated.csv", "2019_06_19_000_annotated.csv",                                               "2019_06_19_001_annotated.csv",  "2019_06_26_002_annotated.csv"]
GROUND_TRUTH = ["2019_06_26_002_annotated.csv","2019_06_13_002_annotated.csv"]

TEST_OBJECT_DETECTIONS = ["2019_06_13_000.csv"]
TEST_GAZE_POSITIONS = ["2019_06_13_000_gaze_positions.csv"]
TEST_GROUND_TRUTH = ["2019_06_13_000_annotated.csv"]


FILE_CHUNK = 900

# CV related
NUMBER_OF_CV_FOLDS = 5    # k index for kfold cross validation


REV_LABELS = {
          #  1: "card",
            2: "face",
            3: "dice",
            4: "key",
            5: "map",
            6: "ball",
            0: "none",

            }

LABELS = {"card": 0,
          "face": 2,
          "dice": 3,
          "key": 4,
          "map": 5,
          "ball": 6,
          "none": 0,
          "uncertain": 0}

REV_INOUTLABELS = {      # This doesn't look good, maybe because the labels are not enough
                0 : "O",
                1 : "B",
                2 : "I"

}

INOUTLABELS = {
                    "O" : 0,
                    "B" : 1,
                    "I" : 2
}

REV_WITHOBJ = {
                1:"B-face",
                2:"I-face",
                3:"B-dice",
                4:"I-dice",
                5:"B-key",
                6:"I-key",
                7:"B-map",
                8:"I-map",
                9:"B-ball",
                10:"I-ball",
                # 11:"B-card",
                # 12:"I-card",
                #
                0:"O"
}
WITHOBJ = {

            "B-face":1,
            "I-face":2,
            "B-dice":3,
            "I-dice":4,
            "B-key":5,
            "I-key":6,
            "B-map":7,
            "I-map":8,
            "B-ball":9,
            "I-ball":10,
            "B-card":0,
            "I-card":0,
            "B-uncertain":0,
            "I-uncertain":0,
            "O":0
}
