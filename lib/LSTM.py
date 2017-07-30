#!/usr/bin/env python
import glob
import os
import pickle
import sys
import time
import urllib

import tensorflow as tf
import tflearn
from tflearn.data_utils import textfile_to_semi_redundant_sequences, \
    random_sequence_from_textfile


class LSTM:
    def __init__(self, model, seq_maxlen=0):
        tf.logging.set_verbosity(tf.logging.FATAL)
        self.model = model
        self.training_data = os.path.join('data', model + '.txt')
        self.model_path = os.path.join('models', model)
        self.checkpoint_path = os.path.join('models', model, model)
        self.seq_maxlen = seq_maxlen

        # Create checkpoint path if it doesn't exist...
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.load_models()

    def train(self, iterations=10000, test_seq_len=100):
        X, Y, char_idx = textfile_to_semi_redundant_sequences(
            self.training_data,
            seq_maxlen=self.seq_maxlen,
            redun_step=3)

        self.save_params([self.seq_maxlen, char_idx])

        m = self.model_arch(dictionary=char_idx)

        for i in range(iterations):
            seed = random_sequence_from_textfile(self.training_data,
                                                 self.seq_maxlen)
            m.fit(X,
                  Y,
                  validation_set=0.1,
                  batch_size=128,
                  n_epoch=1,
                  run_id=self.model)

            print 'Testing with temperature = 1\n--'
            test = m.generate(test_seq_len,
                              temperature=1.0,
                              seq_seed=seed,
                              display=True) + '\n'

            with open("./logs/" + self.model + '.log', "a") as handle:
                handle.write('--\nIteration = ' + str(i + 1) + '\n--\n')
                handle.write(test)
                handle.write('\n\n')

        print('Finished.')

    def generate(self, seed, seq_len=600, temperature=1.0, display=True):
        self.seq_maxlen, char_idx = self.load_params()
        self.seq_maxlen = len(seed)

        m = self.model_arch(dictionary=char_idx)
        m.load(self.get_latest_model())
        
        return m.generate(seq_length=seq_len,
                          temperature=temperature,
                          seq_seed=seed,
                          display=display)

    def model_arch(self,
                   dictionary,
                   activation='softmax',
                   optimizer='adam',
                   loss='categorical_crossentropy',
                   learning_rate=0.001,
                   clip_gradients=5.0,
                   max_checkpoints=7):
        g = tflearn.input_data([None, self.seq_maxlen, len(dictionary)])
        g = tflearn.lstm(g, 512, return_seq=True)
        g = tflearn.dropout(g, 0.5)
        g = tflearn.lstm(g, 512, return_seq=True)
        g = tflearn.dropout(g, 0.5)
        g = tflearn.lstm(g, 512)
        g = tflearn.dropout(g, 0.5)
        g = tflearn.fully_connected(g, len(dictionary), activation=activation)

        g = tflearn.regression(g,
                               optimizer=optimizer,
                               loss=loss,
                               learning_rate=learning_rate)

        m = tflearn.SequenceGenerator(g,
                                      dictionary=dictionary,
                                      seq_maxlen=self.seq_maxlen,
                                      clip_gradients=clip_gradients,
                                      max_checkpoints=max_checkpoints,
                                      checkpoint_path=self.checkpoint_path)

        return m

    def get_latest_model(self):
        files = glob.glob(os.path.join('models', self.model, '*.index'))
        files.sort(reverse=True)
        return files[0].replace('.index', '')

    def save_params(self, params):
        pkl = os.path.join(self.model_path, 'char_idx.pkl')
        with open(pkl, 'wb') as handle:
            pickle.dump(params,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def load_params(self):
        pkl = os.path.join(self.model_path, 'char_idx.pkl')
        with open(pkl, 'rb') as handle:
            return pickle.load(handle)

    def load_models(self):
        if os.path.isfile(self.training_data):
            return

        models = {
            'oxford_dictionary': 'https://raw.githubusercontent.com/sujithps/Dictionary/master/Oxford%20English%20Dictionary.txt',
            'shakespeare': 'https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/shakespeare_input.txt',
            'obama': 'https://raw.githubusercontent.com/samim23/obama-rnn/master/input.txt'
        }

        for key, value in models.iteritems():
            if not self.model == key:
                continue

            print 'Downloading {}'.format(self.model)
            urllib.urlretrieve(value, self.training_data, self.reporthook)
            print '\nDownload complete'

    def reporthook(self, count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                         (percent, progress_size / (1024 * 1024), speed,
                          duration))
        sys.stdout.flush()
