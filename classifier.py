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
            # print(os.path.join(subdir, file))
            filepath = subdir + os.sep + file
            if filepath.endswith(".tsv"):
                print(filepath)
                label = filepath.split(os.sep)[-2]
                labels.append(label)
                p = main(args, filepath)
                p_tcrs.append(p)
    # plt.show()
    neg_p = [per for k, per in enumerate(p_tcrs) if labels[k] == 'CMV+']
    pos_p = [per for k, per in enumerate(p_tcrs) if labels[k] == 'CMV-']
    for all_probs in pos_p:
        plt.hist(all_probs, alpha=0.5, bins=50, label='CMV+')
    for all_probs in neg_p:
        plt.hist(all_probs, alpha=0.5, bins=50, label='CMV-')
    plt.legend()
    plt.show()
    # plt.savefig('cmv_active_tcrs.png')
