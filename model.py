import pycrfsuite
import os
import pickle

from crfsuite_data import prepare_data

with open(os.path.join("data/out", "new_train.pkl"), "rb") as f:
    train = pickle.load(f)
#
# trainer = pycrfsuite.Trainer(algorithm = 'pa',verbose=True)
#
# trainer.set_params({'type':1,
#                     'c': 0.1,
#                     'max_iterations': 1000,
#                     'feature.possible_transitions': False,
#                     'feature.possible_states': False
#                    })
trainer = pycrfsuite.Trainer(verbose=True)

trainer.set_params({'c1': 0.1,   # coefficient for L1 penalty
                    'c2': 0.1,  # coefficient for L2 penalty
                    'max_iterations': 1000,  # stop earlier
                    'feature.possible_transitions': False,
                    'feature.possible_states': False
                   })

for i, data in enumerate(train):
    temp = prepare_data(data)
    for features, ylabel in temp:
        trainer.append(features, ylabel)
    trainer.train("exp_{}".format(i))
    print("Model {} Trained".format(i))

# temp = prepare_data(train[4])
# for features, ylabel in temp:
#     trainer.append(features, ylabel)
# trainer.train("exp_{}".format(4))
# print("Model {} Trained".format(4))

# #
