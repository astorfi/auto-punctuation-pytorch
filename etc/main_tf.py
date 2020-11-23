_, _, files = list(os.walk("./engadget_data"))[0]
"number of files: ", len(files)

import data

fig = plt.figure(figsize=(12, 2))
plt.subplot(131)
plt.plot([len(src) for fn, src in data.source_gen()], linewidth=3, alpha=.7)
plt.title("Distribution of Document Length")
plt.xlabel('Document Index')
plt.ylabel('Document Length')
plt.subplot(132)
plt.plot([len(src) for fn, src in data.validation_gen()], linewidth=3, alpha=.7)
plt.title("Validation Set")
plt.xlabel('Document Index')
plt.ylabel('Document Length')
plt.subplot(133)
plt.plot([len(src) for fn, src in data.test_gen()], linewidth=3, alpha=.7)
plt.title("Test Set")
plt.xlabel('Document Index')
plt.ylabel('Document Length')

plt.tight_layout()
plt.show()

import math, numpy as np, matplotlib.pyplot as plt


input_chars = list(" \nabcdefghijklmnopqrstuvwxyz01234567890")
output_chars = ["<nop>", "<cap>"] + list(".,;:?!\"'$")
import data

i, o = data.extract_punc("ATI'd. I'm not sure if $10 is enough. ", input_chars, output_chars)
print("Punc-less Text:\n========================\n", i)
print("\nPunctuation Operators Extracted:\n========================\n", o)
result = data.apply_punc("".join(i), o)
print("\nVarify that it works by recovering the original string:\n========================\n", result)

import metric


import utils, data, metric, model
from tqdm import tqdm
import numpy as np
from IPython.display import HTML, clear_output
from termcolor import cprint, colored as c

batch_size = 128

char2vec = utils.Char2Vec(chars=input_chars, add_unknown=True)
output_char2vec = utils.Char2Vec(chars=output_chars)
input_size = char2vec.size
output_size = output_char2vec.size

cprint("input_size is: " + c(input_size, 'green') + "; ouput_size is: " + c(output_size, 'green'))
hidden_size = input_size
layers = 1


class GRU(tf.keras.Model):
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

        self.encoder = tf.keras.layers.Dense(
            hidden_size, activation=None, use_bias=True)

        self.forward_gru = tf.keras.layers.GRU(
                    hidden_size, activation='tanh', recurrent_activation='sigmoid', use_bias=True, go_backwards=False
            )

        self.backward_gru = tf.keras.layers.GRU(
            hidden_size, activation='tanh', recurrent_activation='sigmoid', use_bias=True, go_backwards=True
        )

        # Make it bi-directional
        self.gru = tf.keras.layers.Bidirectional(layer=forself.forward_gru, backward_layer=self.backward_gru, merge_mode='concat')

        self.decoder = tf.keras.layers.Dense(
            output_size * self.bi_mul, activation=None, use_bias=True)

    def call(self, x, hidden):
        embeded = x
        gru_output, hidden = self.gru(embeded, hidden.view(self.layers * self.bi_mul, -1, self.hidden_size))
        output = self.decoder(gru_output.contiguous().view(-1, self.hidden_size * self.bi_mul))
        return output.view(self.batch_size, -1, self.output_size), hidden


def Model():
    # Embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        VOCAB_SIZE, hidden_size, embeddings_initializer='uniform',
        embeddings_regularizer=None, activity_regularizer=None,
        embeddings_constraint=None, mask_zero=False, input_length=num_steps)

    # Lstm
    GRU = tf.keras.layers.GRU(
        units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
        kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
        bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        recurrent_constraint=None, bias_constraint=None, dropout=0.0,
        recurrent_dropout=0.0, implementation=2, return_sequences=False,
        return_state=False, go_backwards=False, stateful=False, unroll=False,
        time_major=False, reset_after=True)

    # For classification
    dense_layer = tf.keras.layers.Dense(len(TARGETS))

    # Input
    inputs = tf.keras.layers.Input(shape=[num_steps])
    x = inputs

    # Layers
    x = embedding_layer(x)
    x = lstm(x)
    x = dense_layer(x)

    outputs = x
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = Model()
