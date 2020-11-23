import visdom
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch

# Some parameter
BATCH_TO_SHOW_ACCURACY = 25
BATCH_TO_SHOW_PREDICTION = 100

import utils, data, metric, model
from tqdm import tqdm
import numpy as np
from IPython.display import HTML, clear_output

input_chars = list(" \nabcdefghijklmnopqrstuvwxyz01234567890")
output_chars = ["<nop>", "<cap>"] + list(".,?!")

# torch.set_num_threads(8)
batch_size = 64

char2vec = utils.Char2Vec(chars=input_chars, add_unknown=True)
output_char2vec = utils.Char2Vec(chars=output_chars)
input_size = char2vec.size
output_size = output_char2vec.size

print("input_size is: " + str(input_size) + "; ouput_size is: " + str(output_size))
hidden_size = input_size
layers = 1

# Implement mini-batch
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=1, layers=1, bi=False):
        """
        IMPORTANT: Use batch_first convention for ease of use.
                   However, the hidden layer still use batch middle convension.
        """
        super(GruRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.layers = layers
        self.bi_mul = 2 if bi else 1

        self.encoder = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, self.layers, bidirectional=bi, batch_first=True)
        self.decoder = nn.Linear(hidden_size * self.bi_mul, output_size)
        self.softmax = F.softmax

    def forward(self, x, hidden):
        embeded = x
        gru_output, hidden = self.gru(embeded, hidden.view(self.layers * self.bi_mul, -1, self.hidden_size))
        output = self.decoder(gru_output.contiguous().view(-1, self.hidden_size * self.bi_mul))
        return output.view(self.batch_size, -1, self.output_size), hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.layers * self.bi_mul, self.batch_size, self.hidden_size))

model = Model(input_size, hidden_size, output_size, batch_size=batch_size, layers=layers, bi=True)

learning_rate = 0.5e-2
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer.zero_grad()
loss_fn = nn.CrossEntropyLoss()


seq_length = 500

for epoch_num in range(24):

    losses = []

    for batch_ind, (max_len, sources) in enumerate(tqdm(data.batch_gen(data.train_gen(), batch_size))):

        # prepare the input and output chunks
        input_srcs = []
        punc_targs = []
        for chunk in sources:
            input_source, punctuation_target = data.extract_punc(chunk, char2vec.chars, output_char2vec.chars)
            input_srcs.append(input_source)
            punc_targs.append(punctuation_target)

        # Initialize loss (batch loss)
        loss = 0

        # Initialize hidden
        hidden = model.init_hidden()
        seq_len = data.fuzzy_chunk_len(max_len, seq_length)
        for input_, target_ in zip(zip(*[data.chunk_gen(seq_len, src) for src in input_srcs]),
                                   zip(*[data.chunk_gen(seq_len, tar, ["<nop>"]) for tar in punc_targs])):

            # Reset gradients
            optimizer.zero_grad()

            # Get the embedding layer
            embeded = Variable(char2vec.one_hot_batch(input_))

            # Forward pass to the model
            output, hidden = model(embeded, hidden)

            # Get the target variables
            target_vec = Variable(output_char2vec.char_code_batch(target_))

            # Calculate loss
            new_loss = loss_fn(output.view(-1, model.output_size), target_vec.view(-1))
            loss += new_loss

        # Backward computation for the batch
        loss.backward()
        optimizer.step()

        # Losses indicate the batch losses stored in a list
        losses.append(loss.cpu().data.numpy())


        if (batch_ind + 1) % BATCH_TO_SHOW_ACCURACY == 0:
            print('Epoch {:d} Batch {}'.format(epoch_num + 1, batch_ind + 1))
            print("=================================")
            temperature = 1
            softmax = model.softmax(output.view(-1, model.output_size) / temperature, 1
                                              ).view(model.batch_size, -1, model.output_size)
            indexes = torch.multinomial(softmax.view(-1, model.output_size), 1
                                        ).view(model.batch_size, -1)
            punctuation_output = output_char2vec.vec2list_batch(indexes)

            metric.print_pc(utils.flatten(punctuation_output), utils.flatten(target_))
            print('\n')

        if (batch_ind + 1) % BATCH_TO_SHOW_PREDICTION == 0:
            validate_target = data.apply_punc(input_[0], target_[0])
            result = data.apply_punc(input_[0],
                                     punctuation_output[0])
            print(validate_target)
            print(result)
