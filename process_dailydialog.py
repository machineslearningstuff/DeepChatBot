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
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                seqs = line.split('__eou__')[:-1]
                
                # For each sentence in the line clean and tokenize
                for seq in seqs:
                    seq = clean(seq)
                    list_words = nltk.word_tokenize(seq)
                    words.extend(list_words)
    words = Counter(words)
    print(f'[!] whole vocab size: {len(words)}')
    
    words = words.most_common(cutoff)
    
    # Special tokens
    words.extend([('<sos>', 1), ('<eos>', 1), 
                  ('<unk>', 1), ('<pad>', 1),])
    w2idx = {item[0]:idx for idx, item in enumerate(words)}
    idx2w = [item[0] for item in words]
    with open(vocab, 'wb') as f:
        pkl.dump((w2idx, idx2w), f)
    print(f'[!] Save the vocab into {vocab}, vocab_size: {len(w2idx)}')    

    
    
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

if __name__ == "__main__":
    main()
    



