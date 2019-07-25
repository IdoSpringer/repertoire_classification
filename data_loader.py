import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.autograd as autograd
import os
import csv
from sklearn.utils import shuffle


class TCR(Dataset):
    """TCR dataset, memory or naive"""

    def __init__(self, naive_dir, memory_dir):
        # Word to index dictionary
        amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
        self.amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
        self.naive_dir = naive_dir
        self.memory_dir = memory_dir
        self.labels = []
        self.tcrs = []
        # Read data from directory
        # naive_tcrs = []
        for file in os.listdir(self.naive_dir):
            filename = os.fsdecode(file)
            if filename.endswith(".csv"):
                with open(self.naive_dir + '/' + filename, 'r') as csv_file:
                    csv_file.readline()
                    csv_ = csv.reader(csv_file)
                    for row in csv_:
                        if row[1] == 'control':
                            tcr = row[-1]
                            # naive_tcrs.append(tcr)
                            self.tcrs.append(tcr)
                            self.labels.append('naive')
        # memory_tcrs = []
        for file in os.listdir(self.memory_dir):
            filename = os.fsdecode(file)
            is_memory = 'CM' in filename or 'EM' in filename
            if filename.endswith(".cdr3") and 'beta' in filename and is_memory:
                with open(self.memory_dir + '/' + filename, 'r') as file:
                    for row in file:
                        row = row.strip().split(',')
                        tcr = row[0]
                        # memory_tcrs.append(tcr)
                        self.tcrs.append(tcr)
                        self.labels.append('memory')
            elif filename.endswith(".cdr3") and 'beta' in filename and 'naive' in filename:
                with open(self.memory_dir + '/' + filename, 'r') as file:
                    for row in file:
                        row = row.strip().split(',')
                        tcr = row[0]
                        # naive_tcrs.append(tcr)
                        self.tcrs.append(tcr)
                        self.labels.append('naive')
        # self.tcr_list = naive_tcrs + memory_tcrs
        assert len(self.tcrs) == len(self.labels)
        shuffle(self.tcrs, self.labels)

    def __len__(self):
        return len(self.tcrs)

    def __getitem__(self, idx):
        return self.tcrs[idx], self.labels[idx]


class TCR_Repertoire(Dataset):

    def __init__(self, tsv_file, peptide=None):
        self.peptide = peptide
        self.tcrs = []
        with open(tsv_file, 'r', encoding='unicode_escape') as file:
            file.readline()
            reader = csv.reader(file, delimiter='\t')
            for line in reader:
                tcr = line[1]
                rearrangement_type = line[3]
                templates = line[4]
                if rearrangement_type != 'VDJ':
                    continue
                if tcr is '':
                    continue
                if any(key in tcr for key in ['#', '*', '~', 'O', '/']):
                    continue
                if any(key.islower() for key in tcr):
                    continue
                self.tcrs.append((tcr, templates))

    def __len__(self):
        return len(self.tcrs)

    def __getitem__(self, idx):
        return self.tcrs[idx]


class PeptideRepertoire(Dataset):

    def __init__(self, tsv_file, peptide=None):
        self.peptide = peptide
        self.pairs = []
        with open(tsv_file, 'r', encoding='unicode_escape') as file:
            file.readline()
            reader = csv.reader(file, delimiter='\t')
            for line in reader:
                #tcr = line[1]
                #rearrangement_type = line[3]
                #templates = line[4]
                tcr = line[0]
                templates = line[1]
                if tcr is '':
                    continue
                if any(key in tcr for key in ['#', '*', '~', 'O', '/']):
                    continue
                if any(key.islower() for key in tcr):
                    continue
                self.pairs.append((tcr, templates, self.peptide))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def lstm_convert_seqs(seqs):
    conv = seqs.copy()
    for i in range(len(conv)):
        if any(letter.islower() for letter in conv[i]):
            print(conv[i])
            raise KeyError
        conv[i] = [amino_to_ix[amino] for amino in conv[i]]
    return conv


def lstm_pad_batch(seqs):
    """
    Pad a batch of sequences (part of the way to use RNN batching in PyTorch)
    """
    _seqs = lstm_convert_seqs(seqs)
    # Tensor of sequences lengths
    lengths = torch.LongTensor([len(seq) for seq in _seqs])
    # The padding index is 0 (assumption)
    # Batch dimensions is number of sequences * maximum sequence length
    longest_seq = max(lengths)
    batch_size = len(_seqs)
    # Pad the sequences. Start with zeros and then fill the true sequence
    padded_seqs = autograd.Variable(torch.zeros((batch_size, longest_seq))).long()
    for i, seq_len in enumerate(lengths):
        seq = _seqs[i]
        padded_seqs[i, 0:seq_len] = torch.LongTensor(seq[:seq_len])
    # Return padded batch and the true lengths
    return padded_seqs, lengths


class LSTMBatch:

    def __init__(self, pairs):
        self.tcrs = [pair[0] for pair in pairs]
        self.temps = [pair[1] for pair in pairs]
        self.peps = [pair[2] for pair in pairs]
        self.padded_tcrs, self.tcrs_lengths = lstm_pad_batch(self.tcrs)
        self.padded_peps, self.peps_lengths = lstm_pad_batch(self.peps)

    def to_device(self, device):
        self.padded_tcrs = self.padded_tcrs.to(device)
        self.tcrs_lengths = self.tcrs_lengths.to(device)
        self.padded_peps = self.padded_peps.to(device)
        self.peps_lengths = self.peps_lengths.to(device)


def ae_pad_tcr(tcr, max_len):
    padding = torch.zeros(max_len, 20 + 1)
    _tcr = tcr + 'X'
    for i in range(len(_tcr)):
        amino = _tcr[i]
        padding[i][tcr_atox[amino]] = 1
    return padding


# tcrs must have 21 one-hot, not 22. padding index in pep must be 0.
def ae_convert_tcrs(tcrs, max_len):
    conv = tcrs.copy()
    for i in range(len(tcrs)):
        conv[i] = ae_pad_tcr(tcrs[i], max_len)
    return conv


class AEBatch:

    def __init__(self, pairs, max_len, batch_size):
        self.tcrs = [pair[0] for pair in pairs]
        self.temps = [pair[1] for pair in pairs]
        self.peps = [pair[2] for pair in pairs]
        if len(pairs) != batch_size:
            self.tcr_tensor = None
            self.padded_peps = None
            self.peps_lengths = None
            return
        onehot_tcrs = ae_convert_tcrs(self.tcrs, max_len)
        self.tcr_tensor = torch.zeros((batch_size, max_len, 21))
        for i in range(batch_size):
            self.tcr_tensor[i] = onehot_tcrs[i]
        self.padded_peps, self.peps_lengths = lstm_pad_batch(self.peps)

    def to_device(self, device):
        self.tcr_tensor = self.tcr_tensor.to(device)
        self.padded_peps = self.padded_peps.to(device)
        self.peps_lengths = self.peps_lengths.to(device)


def wrapper(pairs, key, max_len=None, batch_size=None):
    if key == 'lstm':
        return LSTMBatch(pairs)
    elif key == 'ae':
        return AEBatch(pairs, max_len, batch_size)
    pass


# Word to index dictionary
amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
global amino_to_ix
amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
global tcr_atox, pep_atox
tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}
pep_atox = amino_to_ix


def check():
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    global amino_to_ix
    amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    global tcr_atox, pep_atox
    tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}
    pep_atox = amino_to_ix
    global max_len
    max_len = 28
    # Load data
    dataset = PeptideRepertoire('data/CMV+/HIP00594.tsv', peptide='NLVPMVATV')
    print(len(dataset))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    key = 'ae'
    collate_wrapper = lambda x: wrapper(x, key, max_len=max_len)
    train_loader = DataLoader(
        train_set, batch_size=100, shuffle=True, collate_fn=collate_wrapper,
        num_workers=20, pin_memory=True, sampler=None)
    test_loader = DataLoader(
        test_set, batch_size=100, shuffle=True, collate_fn=collate_wrapper,
        num_workers=20, pin_memory=True, sampler=None)
    for b in train_loader:
        print(b.tcrs)
        exit()
    pass


if __name__ == '__main__':
    check()
