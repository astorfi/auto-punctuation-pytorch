import visdom
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import utils, data, metric
from tqdm import tqdm
import numpy as np

""" # Some parameter
"""
BATCH_TO_SHOW_ACCURACY = 25
BATCH_TO_SHOW_PREDICTION = 100
batch_size = 64
NUM_EPOCHS = 25

input_chars = list(" \nabcdefghijklmnopqrstuvwxyz01234567890")
output_chars = ["<nop>", "<cap>"] + list(".,?!")

char2vec = utils.Char2Vec(chars=input_chars, add_unknown=True)
output_char2vec = utils.Char2Vec(chars=output_chars)


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
model = Model(input_size, hidden_size, output_size, batch_size=batch_size, num_layers=num_layers,
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


def process_input_target(input_, target_, hidden):
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


seq_length = 500

for epoch_num in range(NUM_EPOCHS):

    losses = []

    for batch_ind, (max_len, sources) in enumerate(tqdm(data.batch_gen(data.train_gen(), batch_size))):

        # Process & prepare
        input_srcs, punc_targs = prepare_input_output(sources)

        # Initialize loss (batch loss)
        loss = 0

        # Initialize hidden
        hidden = model.init_hidden()
        seq_len = data.fuzzy_chunk_len(max_len, seq_length)
        for input_, target_ in zip(zip(*[data.chunk_gen(seq_len, src) for src in input_srcs]),
                                   zip(*[data.chunk_gen(seq_len, tar, ["<nop>"]) for tar in punc_targs])):
            # Reset gradients
            optimizer.zero_grad()

            # Process input target
            output, hidden, target_vec = process_input_target(input_, target_, hidden)

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

        if (batch_ind + 1) % BATCH_TO_SHOW_ACCURACY == 0:
            print('Epoch {:d} Batch {}'.format(epoch_num + 1, batch_ind + 1))
            print("=================================")

            with torch.no_grad():
                max_len, test_sources = next(data.batch_gen(data.test_gen(), batch_size))
                input_srcs_test, punc_targs_test = prepare_input_output(test_sources)
                for input_, target_ in zip(zip(*[data.chunk_gen(seq_len, src) for src in input_srcs_test]),
                                           zip(*[data.chunk_gen(seq_len, tar, ["<nop>"]) for tar in punc_targs_test])):
                    output, hidden, target_vec = process_input_target(input_, target_, hidden)

                # Prediction probabilities
                probs = F.softmax(output.view(-1, model.output_size), dim=1
                                ).view(model.batch_size, -1, model.output_size)

                # Use argmax to extract predicted labels
                indexes = torch.argmax(probs, axis=2)
                # indexes_m = torch.multinomial(probs.view(-1, model.output_size), num_samples=1
                #                             ).view(model.batch_size, -1)
                punctuation_output = output_char2vec.vec2list_batch(indexes)

                metric.print_pc(utils.flatten(punctuation_output), utils.flatten(target_))
                print('\n')

        if (batch_ind + 1) % BATCH_TO_SHOW_PREDICTION == 0:
            validate_target = data.apply_punc(input_[0], target_[0])
            result = data.apply_punc(input_[0],
                                     punctuation_output[0])
            print(validate_target)
            print(result)
