# Pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from termcolor import cprint, colored

# Other libraries needed
from tqdm import tqdm
import numpy as np
import os
import math
import pandas as pd
import random
from sklearn.metrics import precision_recall_fscore_support
import onnx

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
    BATCH_TO_SHOW_ACCURACY=100,
    BATCH_TO_SHOW_PREDICTION=100,
    batch_size=128,
    NUM_EPOCHS=25,
)

# Data path
df_path = os.path.expanduser("data/data.h5")

# Load hdf5 data
data_df = pd.read_hdf(df_path, 'df')
loaded_dataset = list(data_df.text)

# All the text
text = ' '.join(loaded_dataset)

# The unique characters in the file
chars = sorted(set(text))
print('{} unique characters'.format(len(chars)))

# Desired characters as punctuations
input_chars = list("abcdefghijklmnopqrstuvwxyz01234567890") + [" "]
output_chars = ["<nop>", "<cap>", ".", ",", "?", "!"]

class CharMap():
    def __init__(self, chars=None, add_unknown=False):
        self.chars = chars
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = np.array(chars)
        self.size = len(self.chars)
        self.add_unknown = add_unknown
        if add_unknown:
            self.char2idx['<unk>'] = self.size
            self.size += 1
    def get_ind(self, char):
        try:
            return self.char2idx[char]
        except KeyError:
            if self.add_unknown is False:
                raise KeyError('character is not in dictionary: ' + str([char]))
            return self.char2idx['<unk>']

    def char_code_batch(self, batch):
        return torch.LongTensor([[self.char2idx[char] for char in seq] for seq in batch])

    def vec2list_batch(self, vec):
        chars = [[self.chars[ind] for ind in row] for row in vec.cpu().data.numpy()]
        return chars

char2vec = CharMap(chars=input_chars, add_unknown=True)
output_char2vec = CharMap(chars=output_chars, add_unknown=False)

def add_punctuation(text_input, punctuation):
    assert len(text_input) == len(punctuation), "input string has differnt length from punctuation list" + "".join(
        text_input) + str(punctuation) + str(len(text_input)) + ";" + str(len(punctuation))
    result = ""
    for char1, char2 in zip(text_input, punctuation):
        if char2 == "<cap>":
            result += char1.upper()
        elif char2 == "<nop>":
            result += char1
        else:
            result += char2 + char1
    return result

def extract_punc(string_input, input_chars, output_chars):
    input_source = []
    output_source = []
    input_length = len(string_input)
    i = 0
    while i < input_length:
        char = string_input[i]
        if char.isupper():
            output_source.append("<cap>")
            input_source.append(char.lower())

        if char in output_chars:
            output_source.append(char)
            if i < input_length - 1:
                input_source.append(string_input[i + 1])
            else:
                input_source.append(" ")
            i += 1

        if not char.isupper() and char not in output_chars and char in input_chars:
            input_source.append(char)
            output_source.append("<nop>")

        i += 1
    return input_source, output_source

def prepare_input_output(sources):
    # prepare the input and output chunks
    input_srcs = []
    punc_targs = []
    for chunk in sources:
        input_source, punctuation_target = extract_punc(chunk, char2vec.chars, output_char2vec.chars)
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

def process_char_to_idx(input_, target_):
    # Characters to indexes
    source_shape = [len(input_), len(input_[0])]
    input_ = torch.LongTensor([[[char2vec.get_ind(char)] for char in src] for src in input_])
    input_ = input_[..., 0]

    # Get the target variables
    target_ = Variable(output_char2vec.char_code_batch(target_))
    # u, counts = np.unique(target_vec, return_counts=True)

    return input_, target_

def flatten_(lst):
    # Flattening a nested list
    # Ref: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    return [item for sublist in lst for item in sublist]


class DatasetObject(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sent = self.data[idx]
        sent_len = len(sent)

        # input_source, punctuation_target = data.extract_punc(sent, char2vec.chars, output_char2vec.chars)

        return sent_len, sent


def get_data(dataset, train=True):

    # Calculate len dataset
    data_len = len(dataset)

    # First split to train/test
    if train:
        sub_dataset = dataset[:int(0.8 * data_len)]
    else:
        sub_dataset = dataset[int(0.8 * data_len):]

    # Create the dataset object
    train_or_test = DatasetObject(dataset=sub_dataset)

    return train_or_test


def create_data_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         pin_memory=True, num_workers=0, drop_last=True)
    return loader


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


def make(config):
    # Make the data
    train, test = get_data(loaded_dataset, train=True), get_data(loaded_dataset, train=False)
    train_loader = create_data_loader(train, batch_size=config['batch_size'])
    test_loader = create_data_loader(test, batch_size=config['batch_size'])

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Note: To deal with imbalanced scenario, I used a weighted loss function.
    # The coefficients are arbitrary and the first one is associated with "no punctuation" case.
    # Further details:
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    # https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/8
    weights = torch.tensor([1., 10., 10., 10., 10., 10.])
    criterion = nn.CrossEntropyLoss(weight=weights)

    return model, train_loader, test_loader, criterion, optimizer


def train(model, train_loader, test_loader, criterion, optimizer, config):
    # Run training and track with wandb
    total_batches = len(train_loader) * config['NUM_EPOCHS']
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in range(config['NUM_EPOCHS']):
        for _, (sent_lengths, sources) in enumerate(tqdm(train_loader)):

            loss = train_batch(sent_lengths, sources, model, optimizer, criterion)
            batch_ct += 1

            if batch_ct % config['BATCH_TO_SHOW_ACCURACY'] == 0:
                print('\n Epoch {:d} Batch {}'.format(epoch + 1, batch_ct))
                print("------------------------------")

                with torch.no_grad():
                    # Get data
                    test_sent_lengths, test_sources = next(iter(test_loader))

                    max_len_test = int(max(test_sent_lengths))
                    # seq_len_test = data.fuzzy_chunk_len(max_len_test, seq_length)

                    # Prepare and process
                    input_srcs_test, punc_targs_test = prepare_input_output(test_sources)

                    # Pad sequences to have equal length
                    input_source = _prepare_by_pad(input_srcs_test, max_len_test, filler=[" "])
                    target_punctuation = _prepare_by_pad(punc_targs_test, max_len_test, filler=["<nop>"])

                    # Forward pass
                    input_, target_ = process_char_to_idx(input_source, target_punctuation)

                    # Get the embedding
                    embeded = model.embedding(torch.LongTensor(input_))

                    # Initialize hidden
                    hidden = model.init_hidden()

                    # Forward pass to the model
                    output, hidden = model(embeded, hidden)

                    # Prediction probabilities
                    probs = F.softmax(output.view(-1, model.output_size), dim=1
                                      ).view(config['batch_size'], -1, model.output_size)

                    # Use argmax to extract predicted labels
                    indexes = torch.argmax(probs, axis=2)

                    # Predict punctuation
                    predicted_punctuation = output_char2vec.vec2list_batch(indexes)

                    ############## Evaluation #############

                    # Flatten vectors. Initial size: batch_size,_ as a nested list.
                    pred_to_eval = flatten_(predicted_punctuation)
                    target_to_eval = flatten_(target_punctuation)

                    # Calculate precision_recall_fscore
                    labels = list(set(target_to_eval).union(set(pred_to_eval)))
                    prf = precision_recall_fscore_support(pred_to_eval, target_to_eval, zero_division=0, labels=labels)
                    index = ['precision', 'recall', 'f_score', 'support']
                    df = pd.DataFrame(prf, columns=labels, index=index)

                    # Cut floating points
                    df = df.applymap(lambda x: float('%.2f' % (x)))
                    print('Performance: \n')
                    print(df)
                    print('\n')


            if batch_ct % config['BATCH_TO_SHOW_PREDICTION'] == 0:
                # Add punctuation to the source sentence based on the correct punctuation
                target_text = add_punctuation(input_source[0], target_punctuation[0])

                # Add punctuation to the source sentence based on the predicted punctuation
                predicted_text = add_punctuation(input_source[0],
                                         predicted_punctuation[0])
                cprint('Desired target text: ', 'red', attrs=['bold'])
                print('\t', target_text)
                cprint('Predicted text: ', 'green', attrs=['bold'])
                print('\t', predicted_text)


        #### Save model after each epoch ####
        model_path = 'models/' + "punctuator_" + str(epoch+1) + '.pt'
        torch.save(model, model_path)



def train_batch(sent_lengths, sources, model, optimizer, criterion):

    # Get the max len of sent
    max_len = int(max(sent_lengths))

    # Process & prepare
    input_srcs, punc_targs = prepare_input_output(sources)

    # Initialize hidden
    hidden = model.init_hidden()

    input_ = _prepare_by_pad(input_srcs, max_len, filler=[" "])
    target_ = _prepare_by_pad(punc_targs, max_len, filler=["<nop>"])

    # Reset gradients
    optimizer.zero_grad()

    # Process input target
    input_, target_ = process_char_to_idx(input_, target_)

    # Get the embedding
    embeded = model.embedding(torch.LongTensor(input_))

    # Forward pass to the model
    output, hidden = model(embeded, hidden)

    #### Calculate loss ####
    # Flatten the characters along batches and sequences.
    loss = criterion(output.view(-1, model.output_size), target_.view(-1))

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def test(model, test_loader):
    model.eval()

    # Run the model on some test examples
    cprint('Testing begun: \n', 'blue', attrs=['bold'])
    with torch.no_grad():
        f_score_total, total = 0, 0
        for batch_test_num, (test_sent_lengths, test_sources) in enumerate(tqdm(test_loader)):
            with torch.no_grad():

                # Get size
                total += len(test_sources)

                max_len_test = int(max(test_sent_lengths))
                # seq_len_test = data.fuzzy_chunk_len(max_len_test, seq_length)

                # Prepare and process
                input_srcs_test, punc_targs_test = prepare_input_output(test_sources)

                # Pad sequences to have equal length
                input_source = _prepare_by_pad(input_srcs_test, max_len_test, filler=[" "])
                target_punctuation = _prepare_by_pad(punc_targs_test, max_len_test, filler=["<nop>"])

                # Forward pass
                input_, target_ = process_char_to_idx(input_source, target_punctuation)

                # Get the embedding
                embeded = model.embedding(torch.LongTensor(input_))

                # Initialize hidden
                hidden = model.init_hidden()

                # Forward pass to the model
                output, hidden = model(embeded, hidden)

                # Prediction probabilities
                probs = F.softmax(output.view(-1, model.output_size), dim=1
                                  ).view(config['batch_size'], -1, model.output_size)

                # Use argmax to extract predicted labels
                indexes = torch.argmax(probs, axis=2)

                # Predict punctuation
                predicted_punctuation = output_char2vec.vec2list_batch(indexes)

                ############## Evaluation #############

                # Flatten vectors. Initial size: batch_size,_ as a nested list.
                pred_to_eval = flatten_(predicted_punctuation)
                target_to_eval = flatten_(target_punctuation)

                # Calculate precision_recall_fscore
                labels = list(set(target_to_eval).union(set(pred_to_eval)))
                prf = precision_recall_fscore_support(pred_to_eval, target_to_eval, zero_division=0, labels=labels)
                index = ['precision', 'recall', 'f_score', 'support']
                df = pd.DataFrame(prf, columns=labels, index=index)
                f_score_total += df.loc['f_score', :]

        print(f"F1-score: {100 * f_score_total / (batch_test_num + 1)}%")


def load_pretrained_or_not(model, pretrained=False):

    if pretrained:
        last_saved_epoch = 25
        model_path = 'models/' + "punctuator_" + str(last_saved_epoch) + '.pt'
        model = torch.load(model_path, map_location=torch.device('cpu'))
        return model
    else:
        return model


def model_pipeline(config):

    # make the model, data, and optimization problem
    model, train_loader, test_loader, criterion, optimizer = make(config)

    # and use them to train the model
    train(model, train_loader, test_loader, criterion, optimizer, config)

    # Load pretrained model or use the model at the end of training
    model = load_pretrained_or_not(model, pretrained=True)

    # # and test its final performance
    test(model, test_loader)

    return model

model = model_pipeline(config)
