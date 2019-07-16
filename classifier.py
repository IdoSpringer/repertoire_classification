import argparse
from data_loader import *
from ERGO_models import DoubleLSTMClassifier, PaddingAutoencoder, AutoencoderLSTMClassifier
import os
import matplotlib.pyplot as plt
import numpy as np


def predict(key, model, loader, device):
    model.eval()
    # For all samples
    reacting_tcrs = []
    all_probs = []
    for batch in loader:
        batch.to_device(device)
        # Forward pass
        if key == 'lstm':
            probs = model(batch.padded_tcrs, batch.tcrs_lengths,
                          batch.padded_peps, batch.peps_lengths)
        elif key == 'ae':
            probs = model(batch.tcr_tensor,
                          batch.padded_peps, batch.peps_lengths)
        all_probs.extend(probs.view(1, -1).tolist()[0])
        for i in range(len(batch.tcrs)):
            if probs[i].item() > 0.98:
                # print(batch.tcrs[i])
                reacting_tcrs.append(batch.tcrs[i])
    return np.array(all_probs)


def main(args, data):
    # load repertoire data                                          V
    # load trained ERGO model                                       V
    # Choose specific CMV peptide                                   V
    # get test data batches (repertoire TCR and CMV peptide)        V
    # predict data                                                  V
    # take 1-5% of the TCRs with the highest score from ERGO        V
    # look on the distribution                                      -
    # try to classify repertoire based on distribution (how?)       -

    params = {}
    params['lr'] = 1e-3
    params['wd'] = 1e-5
    params['epochs'] = 200
    params['batch_size'] = 50
    params['lstm_dim'] = 30
    params['emb_dim'] = 10
    params['dropout'] = 0.1
    params['option'] = 0
    params['enc_dim'] = 30
    params['train_ae'] = False
    if args.model_type == 'ae':
        checkpoint = torch.load(args.ae_file)
        params['max_len'] = checkpoint['max_len']
        max_len = checkpoint['max_len']
        params['batch_size'] = checkpoint['batch_size']
    else:
        max_len = None
    # Load data
    dataset = PeptideRepertoire(data, peptide='NLVPMVATV')
    collate_wrapper = lambda x: wrapper(x, args.model_type, max_len=max_len, batch_size=params['batch_size'])
    loader = DataLoader(
        dataset, batch_size=1000, shuffle=True, collate_fn=collate_wrapper,
        num_workers=20, pin_memory=True, sampler=None)
    # Load model
    checkpoint = torch.load(args.model_file)
    if args.model_type == 'ae':
        model = AutoencoderLSTMClassifier(params['emb_dim'], args.device, params['max_len'], 21, params['enc_dim'],
                                          params['batch_size'], args.ae_file, params['train_ae'])
    elif args.model_type == 'lstm':
        model = DoubleLSTMClassifier(params['emb_dim'], params['lstm_dim'], params['dropout'], args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(args.device)
    all_probs = predict(args.model_type, model, loader, args.device)
    high_bin = [p for p in all_probs if p > 0.98]
    return high_bin
    # plt.hist(all_probs, bins=100)
    # plt.show()
    # print(p, len(dataset), p * 100 / len(dataset))
    # return p * 100 / len(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type")
    # parser.add_argument("dataset")
    # parser.add_argument("sampling")
    parser.add_argument("device")
    # parser.add_argument("--protein", action="store_true")
    # parser.add_argument("--hla", action="store_true")
    parser.add_argument("--ae_file", nargs='?', const='pad_full_data_autoencoder_model1.pt',
                        default='pad_full_data_autoencoder_model1.pt')
    # parser.add_argument("--train_auc_file")
    # parser.add_argument("--test_auc_file")
    parser.add_argument("--model_file")
    # parser.add_argument("--roc_file")
    # parser.add_argument("--test_data_file")
    args = parser.parse_args()
    p_tcrs = []
    labels = []
    for subdir, dirs, files in os.walk('data'):
        for file in files:
            filepath = subdir + os.sep + file
            #if filepath.endswith("169.tsv") or filepath.endswith("602.tsv") or\
            #    filepath.endswith("594.tsv") or filepath.endswith("707.tsv"):
            if filepath.endswith(".tsv"):
                print(filepath)
                label = filepath.split(os.sep)[-2]
                labels.append(label)
                p = main(args, filepath)
                p_tcrs.append(p)
    # plt.show()
    neg_p = [per for k, per in enumerate(p_tcrs) if labels[k] == 'CMV-']
    neg_logs = [np.log(1 - np.array(neg_bin)) for neg_bin in neg_p]
    pos_p = [per for k, per in enumerate(p_tcrs) if labels[k] == 'CMV+']
    pos_logs = [-np.log(1 - np.array(pos_bin)) for pos_bin in pos_p]
    for i in range(len(neg_p)):
        plt.hist(neg_logs, label='CMV-', alpha=0.5, density=True, bins=10, histtype='step')
        plt.hist(pos_logs, label='CMV+', alpha=0.5, density=True, bins=10, histtype='step')
        pass
    # plt.hist(neg_logs, label='CMV-', alpha=0.5, stacked=True, density=True, bins=20)
    # plt.hist(pos_logs, label='CMV+', alpha=0.5, stacked=True, density=True, bins=20)
    '''
    for i, (neg_bin, pos_bin), in enumerate(zip(neg_p, pos_p)):
        #print(i, neg_logs[i], pos_logs[i])
        #plt.bar(0, neg_logs[i], bottom=neg_logs[i-1] if i > 0 else 0, width=0.35)
        #plt.bar(1, pos_logs[i], bottom=pos_logs[i-1] if i > 0 else 0, width=0.35)
        weights = np.ones_like(neg_logs[i]) / float(len(neg_logs[i]))
        plt.hist(neg_logs[i], label='CMV-', alpha=0.5, weights=weights, bins=20)
        weights = np.ones_like(pos_logs[i]) / float(len(pos_logs[i]))
        plt.hist(pos_logs[i], label='CMV+', alpha=0.5, weights=weights, bins=20)
    '''
    # plt.xticks(ticks=[0, 1], labels=['CMV-', 'CMV+'])
    # plt.ylabel('Log number of TCRs in repertoire with score > 0.98')
    plt.xlabel('CMV- (left) and CMV+ (right) highest score bin Histograms')
    plt.ylabel('+- log(1 - x) normalized histograms for x > 0.98 scores')
    #plt.legend()
    plt.show()
    # plt.savefig('cmv_active_tcrs.png')
