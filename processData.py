"""
Dataset from: http://hltc.cs.ust.hk/iwslt/index.php/evaluation-campaign/ted-task.html

According to the above link:

"The IWSLT 2012 Evaluation Campaign includes the TED Task, that is the translation of TED Talks, a collection of public speeches on a variety of topics. Three tracks are proposed addressing different research tasks:

    ASR track : automatic transcription of talks from audio to text (in English)
    SLT track: speech translation of talks from audio (or ASR output) to text (from English to French)
    MT track : text translation of talks for two language pairs plus ten optional language pairs."


We used the dataset associated with ASR track.
"""



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

max_len = 500
dataset = []
file = 'data/IWSLT12.TALK.train.en'
with open(file, 'r') as f:
    for line in f:
        # Chunking if len sentence is bigger than something
        if len(line) <= max_len:
            dataset.append(line)
        else:
            num_chunks = math.ceil(len(line) / float(max_len))
            for i in range(num_chunks):
                chunk = line[i * max_len: (i + 1) * max_len]
                dataset.append(chunk)


# Turn to dataframe
df = pd.DataFrame(dataset, columns=['text'])

# Path to save
df_path = os.path.expanduser("~/data/punctuator/data_teds.h5")

# Save as hdf5
df.to_hdf(df_path, key='df', mode='w')