# Pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import utils, data, metric
from torch.utils.data import Dataset, DataLoader

# Other libraries needed
from tqdm import tqdm
import numpy as np
import os
import math
import pandas as pd
import random

# For deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash('setting random seed') % 2 ** 32 - 1)
np.random.seed(hash('To further improve reproducibility') % 2 ** 32 - 1)
torch.manual_seed(hash('Sets a random seed from pytorch random number generators') % 2 ** 32 - 1)
torch.cuda.manual_seed_all(hash('Reproducibility') % 2 ** 32 - 1)

# Set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

""" # Some parameter
"""
config = dict(
    BATCH_TO_SHOW_ACCURACY=25,
    BATCH_TO_SHOW_PREDICTION=100,
    batch_size=128,
    NUM_EPOCHS=25,
)

input_chars = list(" \nabcdefghijklmnopqrstuvwxyz01234567890")
output_chars = ["<nop>", "<cap>"] + list(".,?!")

char2vec = utils.Char2Vec(chars=input_chars, add_unknown=True, add_pad=False)
output_char2vec = utils.Char2Vec(chars=output_chars)

# Data path
df_path = os.path.expanduser("data/data.h5")

# Load hdf5 data
data_df = pd.read_hdf(df_path, 'df')
dataset = list(data_df.text)


class DatasetObject(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset, trainData):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        data_len = len(dataset)
        if trainData:
            self.data = dataset[:int(0.8 * data_len)]
        else:
            self.data = dataset[int(0.8 * data_len):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sent = self.data[idx]
        sent_len = len(sent)

        # input_source, punctuation_target = data.extract_punc(sent, char2vec.chars, output_char2vec.chars)

        return sent_len, sent


""" # Dataset creation
"""
trainData = DatasetObject(dataset=dataset, trainData=True)
train_loader = torch.utils.data.DataLoader(trainData,
                                           batch_size=config['batch_size'], shuffle=True,
                                           num_workers=0, pin_memory=True, drop_last=True)

testData = DatasetObject(dataset=dataset, trainData=False)
test_loader = torch.utils.data.DataLoader(testData,
                                          batch_size=config['batch_size'], shuffle=True,
                                          num_workers=1, drop_last=True)

# Sample from data loaders
sent_sample, sent_len_sample = next(iter(test_loader))


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=1, num_layers=1, bidirectional=False):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, self.num_layers, bidirectional=bidirectional, batch_first=True)
        self.decoder = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x, hidden):
        embeded = x
        gru_output, hidden = self.gru(embeded, hidden.view(self.num_layers * self.num_directions, self.batch_size,
                                                           self.hidden_size))
        output = self.decoder(gru_output.contiguous().view(-1, self.hidden_size * self.num_directions))
        return output.view(self.batch_size, -1, self.output_size), hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size))


""" # Model
Here we initialize the model with its associated parameters.
"""
# How many GRU layers to be stacked
num_layers = 2
input_size = char2vec.size
output_size = output_char2vec.size
hidden_size = 32
bidirectional = True

# Model initialization
model = Model(input_size, hidden_size, output_size, batch_size=config['batch_size'], num_layers=num_layers,
              bidirectional=bidirectional)

""" # Optimizer & Loss
"""
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Note: To deal with imbalanced scenario, I used a weighted loss function.
# The coefficients are arbitrary and the first one is associated with "no punctuation" case.
# Further details:
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/8
weights = torch.tensor([1., 10., 10., 10., 10., 10.])
loss_weighted = nn.CrossEntropyLoss(weight=weights)


def prepare_input_output(sources):
    # prepare the input and output chunks
    input_srcs = []
    punc_targs = []
    for chunk in sources:
        input_source, punctuation_target = data.extract_punc(chunk, char2vec.chars, output_char2vec.chars)
        input_srcs.append(input_source)
        punc_targs.append(punctuation_target)

    return input_srcs, punc_targs


def _prepare_by_pad(sents, max_len, filler):
    padded_seq = []
    for sent in sents:
        s_l = len(sent)
        b_n = math.ceil(s_l / max_len)
        s_pad = sent + filler * (b_n * max_len - s_l)
        padded_seq.append(s_pad)
    return padded_seq


def process_input_foward_pass(input_, target_, hidden):
    # Characters to indexes
    source_shape = [len(input_), len(input_[0])]
    char_to_idx = torch.LongTensor([[[char2vec.get_ind(char)] for char in src] for src in input_])
    char_to_idx = char_to_idx[..., 0]

    # Get the embedding
    embeded = model.embedding(torch.LongTensor(char_to_idx))

    # Forward pass to the model
    output, hidden = model(embeded, hidden)

    # Get the target variables
    target_vec = Variable(output_char2vec.char_code_batch(target_))
    # u, counts = np.unique(target_vec, return_counts=True)

    return output, hidden, target_vec


for epoch in range(config['NUM_EPOCHS']):

    # Create empty batch loss
    losses = []

    for batch_i, (sent_lengths, sources) in enumerate(tqdm(train_loader)):

        # Get the max len of sent
        max_len = int(max(sent_lengths))

        # Process & prepare
        input_srcs, punc_targs = prepare_input_output(sources)

        # Initialize loss (batch loss)
        loss = 0

        # Initialize hidden
        hidden = model.init_hidden()

        input_ = _prepare_by_pad(input_srcs, max_len, filler=[" "])
        target_ = _prepare_by_pad(punc_targs, max_len, filler=["<nop>"])

        # Reset gradients
        optimizer.zero_grad()

        # Process input target
        output, hidden, target_vec = process_input_foward_pass(input_, target_, hidden)

        #### Calculate loss ####
        # Flatten the characters along batches and sequences.
        out_reshaped = output.view(-1, model.output_size)
        new_loss = loss_weighted(output.view(-1, model.output_size), target_vec.view(-1))
        loss += new_loss

        # Backward computation for the batch
        loss.backward()
        optimizer.step()

        # Losses indicate the batch losses stored in a list
        losses.append(loss.cpu().data.numpy())

        if (batch_i + 1) % config['BATCH_TO_SHOW_ACCURACY'] == 0:
            print('Epoch {:d} Batch {}'.format(epoch + 1, batch_i + 1))
            print("=================================")

            with torch.no_grad():
                # Get data
                test_sent_lengths, test_sources = next(iter(test_loader))

                max_len_test = int(max(test_sent_lengths))
                # seq_len_test = data.fuzzy_chunk_len(max_len_test, seq_length)

                # Prepare and process
                input_srcs_test, punc_targs_test = prepare_input_output(test_sources)

                # Pad sequences to have equal length
                input_ = _prepare_by_pad(input_srcs_test, max_len_test, filler=[" "])
                target_ = _prepare_by_pad(punc_targs_test, max_len_test, filler=["<nop>"])

                # Forward pass
                output, hidden, target_vec = process_input_foward_pass(input_, target_, hidden)

                # Prediction probabilities
                probs = F.softmax(output.view(-1, model.output_size), dim=1
                                  ).view(config['batch_size'], -1, model.output_size)

                # Use argmax to extract predicted labels
                indexes = torch.argmax(probs, axis=2)

                # punctuation_output
                punctuation_output = output_char2vec.vec2list_batch(indexes)

                metric.print_pc(utils.flatten(punctuation_output), utils.flatten(target_))
                print('\n')

        if (batch_i + 1) % config['BATCH_TO_SHOW_PREDICTION'] == 0:
            validate_target = data.apply_punc(input_[0], target_[0])
            result = data.apply_punc(input_[0],
                                     punctuation_output[0])
            print(validate_target)
            print(result)
