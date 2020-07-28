import torch.nn as nn, torch, copy, tqdm, math
import torch.nn.functional as F
import pickle
import numpy as np
import argparse


from decoder import *



use_cuda = torch.cuda.is_available()


class HSeq2seq(nn.Module):
    def __init__(self, options):
        super(HSeq2seq, self).__init__()
        self.seq2seq = options.seq2seq
        self.utt_enc = UtteranceEncoder(options.vocab_size, options.emb_size, options.ut_hid_size, options)
        self.intutt_enc = InterUtteranceEncoder(options.ses_hid_size, options.ut_hid_size, options)
        self.dec = Decoder(options)

    def forward(self, batch):
        u1, u1_lenghts, u2, u2_lenghts, u3, u3_lenghts = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        if use_cuda:
            u1 = u1.cuda()
            u2 = u2.cuda()
            u3 = u3.cuda()

        if self.seq2seq:
            o1, o2 = self.utt_enc((u1, u1_lenghts)), self.utt_enc((u2, u2_lenghts))
            qu_seq = torch.cat((o1, o2), 2)
            #final_session_o = self.intutt_enc(qu_seq)
            preds, lmpreds = self.dec((qu_seq, u3, u3_lenghts))
        else:
            o1, o2 = self.utt_enc((u1, u1_lenghts)), self.utt_enc((u2, u2_lenghts))
            qu_seq = torch.cat((o1, o2), 1)
            final_session_o = self.intutt_enc(qu_seq)
            preds, lmpreds = self.dec((final_session_o, u3, u3_lenghts))

        return preds, lmpreds


# encode the hidden states of a number of utterances
class InterUtteranceEncoder(nn.Module):
    def __init__(self, hid_size, inp_size, options):
        super(InterUtteranceEncoder, self).__init__()
        self.hid_size = hid_size
        self.rnn = nn.GRU(hidden_size=hid_size, input_size=inp_size,
                          num_layers=1, bidirectional=False, batch_first=True, dropout=options.drp)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hid_size)
        if use_cuda:
            h_0 = h_0.cuda()
        # output, h_n for output batch is already dim 0
        h_o, h_n = self.rnn(x, h_0)
        h_n = h_n.view(x.size(0), -1, self.hid_size)
        return h_n
        
        
# encode each sentence utterance into a single vector
class UtteranceEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, options):
        super(UtteranceEncoder, self).__init__()
        self.use_embed = options.use_embed
        self.hid_size = hid_size
        self.num_lyr = options.num_lyr
        self.drop = nn.Dropout(options.drp)
        self.direction = 2 if options.bidi else 1
        # by default they requires grad is true
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=10003, sparse=False)
        if self.use_embed:
            pretrained_weight = self.load_embeddings(vocab_size, emb_size)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hid_size,
                          num_layers=self.num_lyr, bidirectional=options.bidi, batch_first=True, dropout=options.drp)

    def forward(self, inp):
        x, x_lengths = inp[0], inp[1]
        bt_siz, seq_len = x.size(0), x.size(1)
        h_0 = torch.zeros(self.direction * self.num_lyr, bt_siz, self.hid_size)
        if use_cuda:
            h_0 = h_0.cuda()
        token_emb = self.embed(x)
        token_emb = self.drop(token_emb)
        token_emb = torch.nn.utils.rnn.pack_padded_sequence(token_emb, x_lengths, batch_first=True)
        gru_out, gru_hid = self.rnn(token_emb, h_0)
        # assuming dimension 0, 1 is for layer 1 and 2, 3 for layer 2
        if self.direction == 2:
            gru_hids = []
            for i in range(self.num_lyr):
                x_hid_temp, _ = torch.max(gru_hid[2*i:2*i + 2, :, :], 0, keepdim=True)
                gru_hids.append(x_hid_temp)
            gru_hid = torch.cat(gru_hids, 0)
        # gru_out, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        # using gru_out and returning gru_out[:, -1, :].unsqueeze(1) is wrong coz its all 0s careful! it doesn't adjust for variable timesteps

        gru_hid = gru_hid[self.num_lyr-1, :, :].unsqueeze(0)
        # take the last layer of the encoder GRU
        gru_hid = gru_hid.transpose(0, 1)

        return gru_hid

    def load_embeddings(self, vocab_size, emb_size):
        vocab_file = './data/word_summary.pkl'
        embed_file = './data/embeddings/glove.840B.300d.txt'
        vocab = {}
        embeddings_index = {}
        with open(vocab_file, 'rb') as fp2:
            dict_data = pickle.load(fp2)

        for x in dict_data:
            tok, f, _, _ = x
            vocab[tok] = f

        #f = open(embed_file)
        with open(embed_file, 'r') as fp:
            f = fp.readlines()
        for line in f:
            values = line.split()
            word = values[0]
            if len(values) > 301:
                continue
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        embedding_matrix = np.zeros((vocab_size, emb_size))
        for word, i in vocab.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
        #print(embedding_vector)
                embedding_matrix[i] = embedding_vector
        print(embedding_matrix.shape)
        return embedding_matrix     
    

if __name__ == "__main__":

    # Testing creating a model
    parser = argparse.ArgumentParser(description='HRED parameter options')
    parser.add_argument('-res_path', dest='res_path', default='./results', help='enter the path in which you want to store the results')
    parser.add_argument('-model_path', dest='model_path', default='./models', help='enter the path in which you want to store the model state')
    parser.add_argument('-e', dest='epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('-pt', dest='patience', type=int, default=-1, help='validtion patience for early stopping default none')
    parser.add_argument('-tf', dest='teacher', action='store_true', default=False, help='default teacher forcing')
    parser.add_argument('-bi', dest='bidi', action='store_true', default=False, help='bidirectional enc/decs')
    parser.add_argument('-test', dest='test', action='store_true', default=False, help='only test or inference')
    parser.add_argument('-shrd_dec_emb', dest='shrd_dec_emb', action='store_true', default=False, help='shared embedding in/out for decoder')
    parser.add_argument('-btstrp', dest='btstrp', default=None, help='bootstrap/load parameters give name')
    parser.add_argument('-lm', dest='lm', action='store_true', default=False, help='enable a RNN language model joint training as well')
    parser.add_argument('-toy', dest='toy', action='store_true', default=False, help='loads only 1000 training and 100 valid for testing')
    parser.add_argument('-pretty', dest='pretty', action='store_true', default=False, help='pretty print inference')
    parser.add_argument('-mmi', dest='mmi', action='store_true', default=False, help='Using the mmi anti-lm for ranking beam')
    parser.add_argument('-s2s', dest='seq2seq', action='store_true', default=False, help='Using baseline seq2seq model')
    parser.add_argument('-drp', dest='drp', type=float, default=0.3, help='dropout probability used all throughout')
    parser.add_argument('-nl', dest='num_lyr', type=int, default=1, help='number of enc/dec layers(same for both)')
    parser.add_argument('-lr', dest='lr', type=float, default=0.001, help='learning rate for optimizer')
    parser.add_argument('-bs', dest='bt_siz', type=int, default=100, help='batch size')
    parser.add_argument('-bms', dest='beam', type=int, default=1, help='beam size for decoding')
    parser.add_argument('-vsz', dest='vocab_size', type=int, default=10004, help='size of vocabulary')
    parser.add_argument('-esz', dest='emb_size', type=int, default=300, help='embedding size enc/dec same')
    parser.add_argument('-uthid', dest='ut_hid_size', type=int, default=600, help='encoder utterance hidden state')
    parser.add_argument('-seshid', dest='ses_hid_size', type=int, default=1200, help='encoder session hidden state')
    parser.add_argument('-dechid', dest='dec_hid_size', type=int, default=600, help='decoder hidden state')
    parser.add_argument('-embed', dest='use_embed', action='store_true', default=False, help='use pretrained word embeddings for the encoder')

    options = parser.parse_args()
    model = HSeq2seq(options)
    
    print(model)











    