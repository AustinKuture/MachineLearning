# -*- coding: utf-8 -*-

import os

import utils
from config import Config
from model import BiRNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

conf = Config()

# wav_files, text_labels = utils.get_wavs_lables()

# words_size, words, word_num_map = utils.create_dict(text_labels)

# bi_rnn = BiRNN(wav_files, text_labels, words_size, words, word_num_map)
# bi_rnn.build_test()

wav_files = ['/Users/microduino/Desktop/RNN/wav/test/1.wav']
txt_labels = ['贝多芬 交响曲 音乐 古典']
words_size, words, word_num_map = utils.create_dict(txt_labels)
bi_rnn = BiRNN(wav_files, txt_labels, words_size, words, word_num_map)
bi_rnn.build_target_wav_file_test(wav_files, txt_labels)



