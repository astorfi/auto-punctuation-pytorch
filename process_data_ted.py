import visdom
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import utils, data, metric
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import math
import pandas as pd


def get_content(fn):
    with open(fn, 'r') as f:
        source = ""
        for line in f:
            source += line

    return source

# Get data
max_len = 500
path = "data/"
dataset = []
child, folders, files = list(os.walk(path))[0]
for fn in files:
    if fn[0] is ".":
        pass
    else:
        src = get_content(path + fn)

        # Chunking if len sentence is bigger than something
        if len(src) <= max_len:
            dataset.append(src)
        else:
            num_chunks = math.ceil(len(src) / float(max_len))
            for i in range(num_chunks):
                chunk = src[i * max_len: (i + 1) * max_len]
                dataset.append(chunk)

# Turn to dataframe
df = pd.DataFrame(dataset, columns=['text'])

# Path to save
df_path = os.path.expanduser("~/data/punctuator/data.h5")

# Save as hdf5
df.to_hdf(df_path, key='df', mode='w')