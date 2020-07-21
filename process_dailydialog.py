import os
import sys

import pickle as pkl

import numpy as np
import argparse
from collections import Counter
import re
import random
from tqdm import tqdm

import nltk
from nltk.util import bigrams
from nltk.util import pad_sequence
from nltk.util import everygrams



def sent_list_debug(sen_list, idx):
    for sent in sen_list[idx]:
        print(sent)
    print('------------------------------')
    


def clean(s):
    # Clean daily dialog dataset sentences, from: https://github.com/gmftbyGMFTBY/MultiTurnDialogZoo/blob/master/utils.py
    s = s.strip().lower()
    s = re.sub(r'(\w+)\.(\w+)', r'\1 . \2', s)
    s = re.sub(r'(\w+)-(\w+)', r'\1 \2', s)
    s = s.replace('。', '.')
    s = s.replace('...', ',')
    s = s.replace(' p . m . ', ' pm ')
    s = s.replace(' P . m . ', ' pm ')
    s = s.replace(' a . m . ', ' am ')
    
    return s    

def generate_vocab(files, vocab, cutoff=50000):
    # Training and validation files, input vocab and output vocab file
    words = []
    max_sent_len = 0
    total_sent_len = 0
    total_num_sent = 0
    
    max_dia_len = 0
    total_num_dia = 0    
    
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                seqs = line.split('__eou__')[:-1]

                # For debugging purposes
                total_num_dia += 1
                if len(seqs) > max_dia_len:
                    max_dia_len = len(seqs)
                
                # For each sentence in the line clean and tokenize
                for seq in seqs:
                    seq = clean(seq)
                    list_words = nltk.word_tokenize(seq)
                    words.extend(list_words)
                    
                    # For debugging purposes
                    total_sent_len += len(list_words)
                    total_num_sent += 1
                    if len(list_words) > max_sent_len:
                        max_sent_len = len(list_words)
                    
    words = Counter(words)
    print(f'[!] whole vocab size: {len(words)}')
    
    # Dialog statistic
    print('Total number of dialogs: %d' %total_num_dia)
    print('Total number of sentences: %d' %total_num_sent)
    print('Num of sent in largest dialog: %d' %max_dia_len)
    print('Average sent per dialog: %.3f' %(total_num_sent/total_num_dia))
    print('Total number of words: %d' %total_sent_len)
    print('Num of words in largest sentence: %d' %max_sent_len)
    print('Average word per sent: %.3f' %(total_sent_len/total_num_sent))
    
    words = words.most_common(cutoff)
    
    # Special tokens
    words.extend([('<sos>', 1), ('<eos>', 1), 
                  ('<unk>', 1), ('<pad>', 1),])
    w2idx = {item[0]:idx for idx, item in enumerate(words)}
    idx2w = [item[0] for item in words]
    with open(vocab, 'wb') as f:
        pkl.dump((w2idx, idx2w), f)
    print(f'[!] Save the vocab into {vocab}, vocab_size: {len(w2idx)}')    


def generate_data(file, c_size = 3, test_ratio = 0.1):
    # Generate training and testing dialog data from given file.
    #   file: contains dialogs per line
    #   c_size: total number of sentences in context + target sentence
    #   ratio:  test/total ratio
    
    data = []

    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            seqs = line.split('__eou__')[:-1]
            clean_seqs = []
            
            # Clean each sentence in the dialog line
            for seq in seqs:
                clean_seqs.append(clean(seq))

            # Each c_size consecutive sentences will be our data
            # c_size - 1 sentences will be the context of the target sentence at c_size
            for i in range(len(seq) - c_size):
                sentSlice = clean_seqs[i:i+c_size]
                if len(sentSlice) != c_size: continue
                data.append(sentSlice)

    print('Total data: %d' %len(data))

    # Debug some random data
    sent_list_debug(data, 156)
    sent_list_debug(data, 1569)
    sent_list_debug(data, 15634)
    
    # Now randomly sample training and testing data to track training process.
    # Each list of sentences in data is one sample of X, y for training where
    # X data[idx][0: end-1] and y is data[idx][end]
    # Binarization will be done later, now only separate sentences and pickle sentences
    perm = np.random.permutation(len(data))
    num_test = int(len(data) * test_ratio)
    test_idx = perm[0:num_test]
    train_idx = perm[num_test:]
    test_data  = [data[t] for t in test_idx]
    train_data = [data[t] for t in train_idx]
    print('Number of train samples: %d' %len(train_data))
    print('Number of test samples: %d' %len(test_data))
    with open('./dailydialog.pkl', 'wb') as f:
        pkl.dump((train_data, test_data), f)
    
    
def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, nargs='+', default=None, help='file for generating the vocab')
    parser.add_argument('--vocab', type=str, default='./vocab.pkl', help='input or output vocabulary')
    parser.add_argument('--cutoff', type=int, default=10000, help='cutoff of the vocabulary')
    parser.add_argument('--ctx', type=int, default=3, help='number of context sentences before target sentence')
    parser.add_argument('--maxlen', type=int, default=50, help='maximum length of sentences')
    
    args = parser.parse_args()
    
    # NLTK requires this
    nltk.download('punkt')
    
    
    # Process vocabulary first if the file doesn't exists
    if not os.path.exists(args.vocab):
        generate_vocab(args.file, args.vocab, cutoff=args.cutoff)

    # Testing 
    with open('./vocab.pkl', 'rb') as f:
        w2idx, idx2w = pkl.load(f)
        print(idx2w[0:10])
        
    if not os.path.exists('./dailydialog.pkl'):
        # Now start generating train and test data sentence lists
        generate_data(args.file[0], args.ctx, 0.1)
        

if __name__ == "__main__":
    main()
    



