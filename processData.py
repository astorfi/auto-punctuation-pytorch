"""
Dataset from: http://hltc.cs.ust.hk/iwslt/index.php/evaluation-campaign/ted-task.html

According to the above link:

"The IWSLT 2012 Evaluation Campaign includes the TED Task, that is the translation of TED Talks, a collection of public speeches on a variety of topics. Three tracks are proposed addressing different research tasks:

    ASR track : automatic transcription of talks from audio to text (in English)
    SLT track: speech translation of talks from audio (or ASR output) to text (from English to French)
    MT track : text translation of talks for two language pairs plus ten optional language pairs."


We used the dataset associated with ASR track.
"""
import os
import pandas as pd
import math
import unicodedata

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

max_len = 500
dataset = []
file = 'data/IWSLT12.TALK.train.en'
with open(file, 'r') as f:
    for line in f:

        # How can I remove a trailing newline (\n)?
        # Ref: https://stackoverflow.com/questions/275018/how-can-i-remove-a-trailing-newline
        line = line.rstrip()

        # Turn a Unicode string to plain ASCII
        line = unicodeToAscii(line.strip())

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
df_path = os.path.expanduser("data/data.h5")

# Save as hdf5
df.to_hdf(df_path, key='df', mode='w')