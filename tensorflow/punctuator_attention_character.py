# -*- coding: utf-8 -*-
"""Punctuator_attention.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1efpbXGrBxqikADcO9OcimK_idEWvitDj
"""

# from google.colab import drive
# drive.mount('/content/drive')

import re
import string
import glob
import os
import random
import collections
import os
import tensorflow as tf
from sklearn.metrics import classification_report
import unicodedata

# /content/drive/MyDrive

# https://www.tensorflow.org/guide/keras/rnn
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough

# Printable characters
printable = set(string.printable)


# Model checkpoint
def check_dir_(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


# Download and unzip data.

import os

os.system("ls -l")

# Download and unzip data.
os.system("mkdir -p /home/sina/punctuator/punctuation/tmp")
os.system("mkdir -p /home/sina/punctuator/punctuation/data")
os.system("mkdir -p /home/sina/punctuator/punctuation/datapreped")
os.system(
    "wget -q -P /home/sina/punctuator/punctuation/tmp/ https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/europarl_raw.zip")
os.system(
    "wget -q -P /home/sina/punctuator/punctuation/tmp/ https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/comtrans.zip")
os.system("unzip -q -o /home/sina/punctuator/punctuation/tmp/europarl_raw.zip -d /home/sina/punctuator/punctuation/tmp")
os.system("unzip -q -o /home/sina/punctuator/punctuation/tmp/comtrans.zip -d /home/sina/punctuator/punctuation/tmp")
os.system("cp /home/sina/punctuator/punctuation/tmp/europarl_raw/english/* /home/sina/punctuator/punctuation/data")

# On English text will be extracted.
sentences = []
with open('/home/sina/punctuator/punctuation/tmp/comtrans/alignment-en-fr.txt', 'r', encoding="ISO-8859-1") as file:
    line_n = 0
    # Write lines 1,4,7,... as they are english
    for sent in file.readlines():
        line_n += 1
        if (line_n - 1) % 3 == 0:
            sentences.append(sent)

print("The total number of sentences: {}".format(len(sentences)))

# Cleaning
clean_sentences = []
i = 0
for sent in sentences:

    # Remove white spaces
    sent = sent.strip()

    # If the sentence is in ()
    bad_condition_1 = sent.endswith(')') and sent.startswith('(')

    # If sentence does not end with any of the .!?
    bad_condition_2 = not sent.endswith('.') and not sent.endswith('!') and not sent.endswith('?')

    # If '...' is in the sentence
    bad_condition_3 = sent.find('...') != -1

    if bad_condition_1 or bad_condition_2 or bad_condition_3:
        continue

    # Remove quotes, apostrophes, leading dashes.
    sent = re.sub('"', '', sent)
    sent = re.sub(' \' s ', 's ', sent)
    sent = re.sub('\'', '', sent)
    sent = re.sub('^- ', '', sent)

    # Clean double punctuations.
    sent = re.sub('\? \.', '\?', sent)
    sent = re.sub('\! \.', '\!', sent)

    # Extract human names to reduce vocab size. There are many names like 'Mrs Plooij-van Gorsel'
    # 'Mr Cox'.
    sent = re.sub('Mr [\w]+ [A-Z][\w]+ ', '[humanname] ', sent)
    sent = re.sub('Mrs [\w]+ [A-Z][\w]+ ', '[humanname] ', sent)
    sent = re.sub('Mr [\w]+ ', '[humanname] ', sent)
    sent = re.sub('Mrs [\w]+ ', '[humanname] ', sent)

    # Remove brackets and contents inside.
    sent = re.sub('\(.*\) ', '', sent)
    sent = re.sub('\(', '', sent)
    sent = re.sub('\)', '', sent)

    # Extract numbers to reduce the vocab size.
    sent = re.sub('[0-9\.]+ ', '[number] ', sent)

    # Replace i.e., p.m., a.m. to reduce confusion on period.
    sent = re.sub(' i\.e\.', ' for example', sent)
    sent = re.sub(' p\.m\.', ' pm', sent)
    sent = re.sub(' a\.m\.', ' am', sent)

    # All lower case
    # sent = sent.lower()

    # If everything above passed, we get the sentence
    clean_sentences.append(sent)

print("The total number of clean sentences: {}".format(len(clean_sentences)))


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(sent):
    status = True

    sent = unicode_to_ascii(sent.strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    sent = re.sub(r"([?.!,¿])", r" \1 ", sent)
    sent = re.sub(r'[" "]+', " ", sent)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sent = re.sub(r"[^a-zA-Z?.!,¿:]+", " ", sent)

    # Again remove space from the begining and end of string
    sent = sent.strip()

    # If the sentence is in ()
    bad_condition_1 = sent.endswith(')') and sent.startswith('(')

    # If sentence does not end with any of the .!?
    bad_condition_2 = not sent.endswith('.') and not sent.endswith('!') and not sent.endswith('?')

    # If '...' is in the sentence
    bad_condition_3 = sent.find('...') != -1

    if bad_condition_1 or bad_condition_2 or bad_condition_3:
        status = False

    # Remove quotes, apostrophes, leading dashes.
    sent = re.sub('"', '', sent)
    sent = re.sub(' \' s ', 's ', sent)
    sent = re.sub('\'', '', sent)
    sent = re.sub('^- ', '', sent)

    # Clean double punctuations.
    sent = re.sub('\? \.', '\?', sent)
    sent = re.sub('\! \.', '\!', sent)

    # Extract human names to reduce vocab size. There are many names like 'Mrs Plooij-van Gorsel'
    # 'Mr Cox'.
    sent = re.sub('Mr [\w]+ [A-Z][\w]+ ', '[humanname] ', sent)
    sent = re.sub('Mrs [\w]+ [A-Z][\w]+ ', '[humanname] ', sent)
    sent = re.sub('Mr [\w]+ ', '[humanname] ', sent)
    sent = re.sub('Mrs [\w]+ ', '[humanname] ', sent)

    # Remove brackets and contents inside.
    sent = re.sub('\(.*\) ', '', sent)
    sent = re.sub('\(', '', sent)
    sent = re.sub('\)', '', sent)

    # Extract numbers to reduce the vocab size.
    sent = re.sub('[0-9\.]+ ', '[number] ', sent)

    # Replace i.e., p.m., a.m. to reduce confusion on period.
    sent = re.sub(' i\.e\.', ' for example', sent)
    sent = re.sub(' p\.m\.', ' pm', sent)
    sent = re.sub(' a\.m\.', ' am', sent)
    
    sent = sent.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    return sent, status


clean_sentences = []
for sent in sentences:
    sent, status = preprocess_sentence(sent)
    if status:
        clean_sentences.append(sent)
    else:
        continue

print("The total number of clean sentences: {}".format(len(clean_sentences)))

print(clean_sentences[-1])

# # Create paragraphs from sentences
# counter = 0
# sent_per_parag = 4
# parapraphs = []
# for i in range(len(clean_sentences) // sent_per_parag):
#   paragraph = clean_sentences[i * sent_per_parag: (i+1) * sent_per_parag]
#
#   # Add '<eos>' to end of each paragraph
#   parapraphs.append(' '.join(paragraph))
#
# par = parapraphs[1]


# The following punctuations are important
PUNCTUATIONS = (u'.', u',', u'?', u'!', u':')

# `n` is no punctuation.
TARGETS = list(PUNCTUATIONS) + ['<nop>']


def process_txt(txt):
    input_source = []
    output_source = []
    words = txt.split()

    idx = 0
    while idx < len(words) - 1:

        if words[idx + 1] in PUNCTUATIONS:
            input_source.append(words[idx])
            output_source.append(words[idx + 1])
            idx += 2

        else:
            input_source.append(words[idx])
            output_source.append('<nop>')
            idx += 1

    input_source = '<start> ' + ' '.join(input_source) + ' <end>'
    output_source = '<start> ' + ' '.join(output_source) + ' <end>'
    return input_source, output_source


# Create datasets

data_words = []
data_puncs = []

for sent in clean_sentences:
    output_source, input_source = process_txt(sent)
    data_words.append(input_source)
    data_puncs.append(output_source)

print(data_words[-1])
print(data_puncs[-1])


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='', lower=False, split=' ', char_level=True, oov_token='<unk>')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')

    return tensor, lang_tokenizer


def load_dataset(targ_lang, inp_lang):
    # creating cleaned input, output pairs

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


# Try experimenting with the size of that dataset
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(data_puncs, data_words)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

sys.exit()

from sklearn.model_selection import train_test_split

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                target_tensor,
                                                                                                test_size=0.2)

# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))


print("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[0])
print()
print("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[0])

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


checkpoint_dir = '/content/drive/MyDrive/punctuator/training_checkpoints'
check_dir_(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


import time

EPOCHS = 50

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    # fontdict = {'fontsize': 14}

    # ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    # ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.show()


import numpy as np


def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    # attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    # plot_attention(attention_plot, sentence.split(' '), result.split(' '))


def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence, status = preprocess_sentence(sentence)

    # print(sentence)
    # print(len(inp_lang.word_index.keys()))
    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

translate(u'The session is over I would like to finish rest and go home')
