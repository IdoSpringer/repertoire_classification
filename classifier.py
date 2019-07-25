import argparse
from data_loader import *
from ERGO_models import DoubleLSTMClassifier, AutoencoderLSTMClassifier
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np
import pickle


def predict(key, model, loader, device):
    model.eval()
    # For all samples
    tcrs = []
    temps = []
    reacting_tcrs = []
    all_probs = []
    for batch in loader:
        if key == 'ae' and batch.tcr_tensor is None:
            continue
        batch.to_device(device)
        # Forward pass
        if key == 'lstm':
            probs = model(batch.padded_tcrs, batch.tcrs_lengths,
                          batch.padded_peps, batch.peps_lengths)
        elif key == 'ae':
            probs = model(batch.tcr_tensor,
                          batch.padded_peps, batch.peps_lengths)
        all_probs.extend(probs.view(1, -1).tolist()[0])
        tcrs.extend(batch.tcrs)
        temps.extend(batch.temps)
        '''
        for i in range(len(batch.tcrs)):
            if probs[i].item() > 0.98:
                # print(batch.tcrs[i])
                reacting_tcrs.append(batch.tcrs[i])
        '''
    return tcrs, temps, np.array(all_probs)


def save_predictions_to_file(repertoire_filename, peptide):
    # all repertoire, some CMV peptides
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
        batch_size = 50
    else:
        max_len = None
        batch_size = 1000
    # Load data
    dataset = PeptideRepertoire(repertoire_filename, peptide=peptide)
    collate_wrapper = lambda x: wrapper(x, args.model_type, max_len=max_len, batch_size=params['batch_size'])
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_wrapper,
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
    tcrs, temps, all_probs = predict(args.model_type, model, loader, args.device)
    assert len(tcrs) == len(all_probs)
    # pathology = repertoire_filename.split("/")[-2]
    rep_id = repertoire_filename.split("/")[-1]
    assert rep_id.endswith('.tsv')
    rep_id = rep_id[:-4]
    with open('/'.join(['ergo_predictions2', args.model_type.upper(),
                        rep_id + '_' + peptide]) + '.pickle', 'wb') as handle:
            pickle.dump(zip(tcrs, temps, all_probs), handle)
    # save predictions as (tcr, score) long list
    # file name is model_type / pathology / repertoire_name + peptide
    pass


def read_predictions_from_file(filepath):
    with open(filepath, 'rb') as handle:
        z = pickle.load(handle)
        tcrs, temps, scores = zip(*z)
    return tcrs, temps, scores


def main(args, data):
    # load repertoire data                                          V
    # load trained ERGO model                                       V
    # Choose specific CMV peptide                                   V
    # get test data batches (repertoire TCR and CMV peptide)        V
    # predict data                                                  V
    # take 1-5% of the TCRs with the highest score from ERGO        V
    # look on the distribution                                      V
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
    all_probs = predict(args.model_type, model, loader, args.device)[1]
    high_bin = [p for p in all_probs if p > 0.98]
    return high_bin
    # plt.hist(all_probs, bins=100)
    # plt.show()
    # print(p, len(dataset), p * 100 / len(dataset))
    # return p * 100 / len(dataset)
    pass


def save_predictions():
    cmv_peps = ['NLVPMVATV', 'VTEHDTLLY', 'TPRVTGGGAM']
    for subdir, dirs, files in os.walk('emerson_tcrs_with_temps'):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".tsv"):
                print(filepath)
                for pep in cmv_peps:
                    save_predictions_to_file(filepath, pep)


def reg_score_hist():
    p_tcrs = []
    labels = []
    neg_p = []
    pos_p = []
    for subdir, dirs, files in os.walk('ergo_predictions'):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith("NLVPMVATV.pickle") and args.model_type in filepath.lower():
                print(filepath)
                label = filepath.split(os.sep)[-2]
                labels.append(label)
                tcrs, preds = read_predictions_from_file(filepath)
                if label == 'CMV-':
                    neg_p.append(preds)
                elif label == 'CMV+':
                    pos_p.append(preds)
                # p_tcrs.append(preds)
    # neg_p = [per for k, per in enumerate(p_tcrs) if labels[k] == 'CMV-']
    # neg_logs = [np.log(1 - np.array(neg_bin)) for neg_bin in neg_p]
    # print(len(neg_logs[0]), len(neg_logs[1]))
    # pos_p = [per for k, per in enumerate(p_tcrs) if labels[k] == 'CMV+']
    # pos_logs = [np.log(1 - np.array(pos_bin)) for pos_bin in pos_p]
    # bins = np.histogram(neg_p[0], density=True, bins='auto')[1]
    # print(bins)
    # neg_hists = [np.histogram(k, density=True, bins=bins)[0] for k in neg_p]
    # pos_hists = [np.histogram(k, density=True, bins=bins)[0] for k in pos_p]
    neg_colors = cm.Reds(np.linspace(0.5, 1, 10))
    pos_colors = cm.Blues(np.linspace(0.5, 1, 10))
    plt.hist(neg_p, stacked=False, color=neg_colors)
    plt.hist(pos_p, stacked=False, color=pos_colors)
    '''
    cmap = cmap = plt.cm.coolwarm
    custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                    Line2D([0], [0], color=cmap(1.), lw=4)]
    fig, ax = plt.subplots()
    for neg_hist, nc, pos_hist, pc in zip(neg_hists, neg_colors, pos_hists, pos_colors):
        ax.plot(range(len(neg_hists[0])), neg_hist, color=nc)
        ax.plot(range(len(pos_hists[0])), pos_hist, color=pc)
    ax.legend(custom_lines, ['CMV+', 'CMV-'])
    plt.xticks([k for k in range(len(pos_hists[0])) if k % 5 == 0],
               [int(b) for i, b in enumerate(bins[:-1]) if i % 5 == 0])
    plt.xlabel('log(1 - x) normalized histograms for x > 0.98 scores')
    plt.title("Highest bin plotted histograms based on ERGO CMV peptide scores")
    # plt.ylabel('+- log(1 - x) normalized histograms for x > 0.98 scores')
    '''
    plt.show()
    pass


def plot_score_histograms():
    # plot histograms
    p_tcrs = []
    labels = []
    neg_p = []
    pos_p = []
    for subdir, dirs, files in os.walk('ergo_predictions'):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith("NLVPMVATV.pickle"):
                print(filepath)
                label = filepath.split(os.sep)[-2]
                labels.append(label)
                tcrs, preds = read_predictions_from_file(filepath)
                if label == 'CMV-':
                    neg_p.append([pred for pred in preds if pred > 0.98])
                elif label == 'CMV+':
                    pos_p.append([pred for pred in preds if pred > 0.98])
                # p_tcrs.append(preds)
    # neg_p = [per for k, per in enumerate(p_tcrs) if labels[k] == 'CMV-']
    neg_logs = [np.log(1 - np.array(neg_bin)) for neg_bin in neg_p]
    print(len(neg_logs[0]), len(neg_logs[1]))
    # pos_p = [per for k, per in enumerate(p_tcrs) if labels[k] == 'CMV+']
    pos_logs = [np.log(1 - np.array(pos_bin)) for pos_bin in pos_p]
    bins = np.histogram(neg_logs[0], density=True, bins='auto')[1]
    print(bins)
    neg_hists = [np.histogram(k, density=True, bins=bins)[0] for k in neg_logs]
    pos_hists = [np.histogram(k, density=True, bins=bins)[0] for k in pos_logs]
    # plt.hist(neg_hists, histtype='step', stacked=True, color=colors)
    # plt.(pos_hists, histtype='step', stacked=True, color=colors)
    neg_colors = cm.Reds(np.linspace(0.5, 1, 10))
    pos_colors = cm.Blues(np.linspace(0.5, 1, 10))

    cmap = cmap = plt.cm.coolwarm
    custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                    Line2D([0], [0], color=cmap(1.), lw=4)]
    fig, ax = plt.subplots()
    for neg_hist, nc, pos_hist, pc in zip(neg_hists, neg_colors, pos_hists, pos_colors):
        ax.plot(range(len(neg_hists[0])), neg_hist, color=nc)
        ax.plot(range(len(pos_hists[0])), pos_hist, color=pc)
    ax.legend(custom_lines, ['CMV+', 'CMV-'])
    plt.xticks([k for k in range(len(pos_hists[0])) if k%5 == 0], [int(b) for i, b in enumerate(bins[:-1]) if i%5 == 0])
    plt.xlabel('log(1 - x) normalized histograms for x > 0.98 scores')
    plt.title("Highest bin plotted histograms based on ERGO CMV peptide scores")
    # plt.ylabel('+- log(1 - x) normalized histograms for x > 0.98 scores')
    plt.show()


def plot_multiple_peps_hists():
    # plot histograms
    cmv_peps = ['NLVPMVATV', 'VTEHDTLLY', 'TPRVTGGGAM']
    for i, pep in enumerate(cmv_peps):
        p_tcrs = []
        labels = []
        neg_p = []
        pos_p = []
        for subdir, dirs, files in os.walk('ergo_predictions'):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(pep + ".pickle") and args.model_type in filepath.lower():
                    print(filepath)
                    label = filepath.split(os.sep)[-2]
                    labels.append(label)
                    tcrs, preds = read_predictions_from_file(filepath)
                    if label == 'CMV-':
                        neg_p.append([pred for pred in preds if pred > 0.98])
                    elif label == 'CMV+':
                        pos_p.append([pred for pred in preds if pred > 0.98])
                    # p_tcrs.append(preds)
        # neg_p = [per for k, per in enumerate(p_tcrs) if labels[k] == 'CMV-']
        neg_logs = [np.log(1 - np.array(neg_bin)) for neg_bin in neg_p]
        print(len(neg_logs[0]), len(neg_logs[1]))
        # pos_p = [per for k, per in enumerate(p_tcrs) if labels[k] == 'CMV+']
        pos_logs = [np.log(1 - np.array(pos_bin)) for pos_bin in pos_p]
        bins = np.histogram(neg_logs[0], density=True, bins='auto', range=(-14.0, -4.0))[1]
        print(bins)
        neg_hists = [np.histogram(k, density=True, bins=bins)[0] for k in neg_logs]
        pos_hists = [np.histogram(k, density=True, bins=bins)[0] for k in pos_logs]
        # plt.hist(neg_hists, histtype='step', stacked=True, color=colors)
        # plt.(pos_hists, histtype='step', stacked=True, color=colors)
        ax = plt.subplot(1, 3, i+1)

        neg_colors = cm.Reds(np.linspace(0.5, 1, 10))
        pos_colors = cm.Blues(np.linspace(0.5, 1, 10))
        cmap = plt.cm.coolwarm
        custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                        Line2D([0], [0], color=cmap(1.), lw=4)]
        # fig, ax = plt.subplots()
        for neg_hist, nc, pos_hist, pc in zip(neg_hists, neg_colors, pos_hists, pos_colors):
            ax.plot(range(len(neg_hists[0])), neg_hist, color=nc)
            ax.plot(range(len(pos_hists[0])), pos_hist, color=pc)
        ax.legend(custom_lines, ['CMV+, ' + pep, 'CMV-, ' + pep])
        ax.set_xticks([k for k in range(len(pos_hists[0])) if k % 5 == 0])
        ax.set_xticklabels([int(b) for i, b in enumerate(bins[:-1]) if i % 5 == 0])
        if i == 1:
            plt.xlabel('log(1 - x) normalized histograms for x > 0.98 scores')
            plt.title("Highest bin plotted histograms based on ERGO-" +args.model_type.upper() + " CMV peptide scores")
    plt.show()
    pass


def score_hists_with_templates():
    # plot histograms
    cmv_peps = ['NLVPMVATV', 'VTEHDTLLY', 'TPRVTGGGAM']
    for i, pep in enumerate(cmv_peps):
        p_tcrs = []
        labels = []
        neg_p = []
        pos_p = []
        for subdir, dirs, files in os.walk('ergo_predictions'):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(pep + ".pickle") and args.model_type in filepath.lower():
                    print(filepath)
                    label = filepath.split(os.sep)[-2]
                    labels.append(label)
                    tcrs, temps, preds = read_predictions_from_file(filepath)
                    # print(temps[:50])
                    # print(preds)
                    product = [[k] * int(c) for k, c in zip(preds, temps) if c != 'null']
                    # print(product[:50])
                    flat = []
                    for l in product:
                        flat.extend(l)
                    preds = flat
                    # print(flat[:50])
                    if label == 'CMV-':
                        neg_p.append([pred for pred in preds if pred > 0.98])
                    elif label == 'CMV+':
                        pos_p.append([pred for pred in preds if pred > 0.98])
                    # p_tcrs.append(preds)
        # neg_p = [per for k, per in enumerate(p_tcrs) if labels[k] == 'CMV-']
        neg_logs = [np.log(1 - np.array(neg_bin)) for neg_bin in neg_p]
        print(len(neg_logs[0]), len(neg_logs[1]))
        # pos_p = [per for k, per in enumerate(p_tcrs) if labels[k] == 'CMV+']
        pos_logs = [np.log(1 - np.array(pos_bin)) for pos_bin in pos_p]
        bins = np.histogram(neg_logs[0], density=True, bins='auto', range=(-14.0, -4.0))[1]
        print(bins)
        neg_hists = [np.histogram(k, density=True, bins=bins)[0] for k in neg_logs]
        pos_hists = [np.histogram(k, density=True, bins=bins)[0] for k in pos_logs]
        # plt.hist(neg_hists, histtype='step', stacked=True, color=colors)
        # plt.(pos_hists, histtype='step', stacked=True, color=colors)
        ax = plt.subplot(1, 3, i + 1)
        neg_colors = cm.Reds(np.linspace(0.5, 1, 10))
        pos_colors = cm.Blues(np.linspace(0.5, 1, 10))
        cmap = plt.cm.coolwarm
        custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                        Line2D([0], [0], color=cmap(1.), lw=4)]
        # fig, ax = plt.subplots()
        for neg_hist, nc, pos_hist, pc in zip(neg_hists, neg_colors, pos_hists, pos_colors):
            ax.plot(range(len(neg_hists[0])), neg_hist, color=nc)
            ax.plot(range(len(pos_hists[0])), pos_hist, color=pc)
        ax.legend(custom_lines, ['CMV+, ' + pep, 'CMV-, ' + pep])
        ax.set_xticks([k for k in range(len(pos_hists[0])) if k % 5 == 0])
        ax.set_xticklabels([int(b) for i, b in enumerate(bins[:-1]) if i % 5 == 0])
        if i == 1:
            plt.xlabel('log(1 - x) normalized histograms for x > 0.98 scores')
            plt.title("Highest bin plotted histograms based on ERGO-" + args.model_type.upper() + " CMV peptide scores")
    plt.show()
    pass


def plot_single_hist():
    # plot histograms
    cmv_peps = ['NLVPMVATV', 'VTEHDTLLY', 'TPRVTGGGAM']
    for i, pep in enumerate([cmv_peps[0]]):
        p_tcrs = []
        labels = []
        neg_p = []
        pos_p = []
        for subdir, dirs, files in os.walk('ergo_predictions'):
            for file in files:
                filepath = subdir + os.sep + file
                correct_hla = 'HIP00594' in filepath or 'HIP00602' in filepath
                if filepath.endswith(pep + ".pickle") and args.model_type in filepath.lower() and correct_hla:
                    print('----------------')
                    print(filepath)
                    label = filepath.split(os.sep)[-2]
                    labels.append(label)
                    tcrs, temps, preds = read_predictions_from_file(filepath)
                    if label == 'CMV-':
                        neg_p.append([pred for pred in preds if pred > 0.98])
                    elif label == 'CMV+':
                        pos_p.append([pred for pred in preds if pred > 0.98])
                    # p_tcrs.append(preds)
        # neg_p = [per for k, per in enumerate(p_tcrs) if labels[k] == 'CMV-']
        print(len(neg_p), len(pos_p))
        neg_logs = [np.log(1 - np.array(neg_bin)) for neg_bin in neg_p]
        # print(len(neg_logs[0]), len(neg_logs[1]))
        # pos_p = [per for k, per in enumerate(p_tcrs) if labels[k] == 'CMV+']
        pos_logs = [np.log(1 - np.array(pos_bin)) for pos_bin in pos_p]
        bins = np.histogram(neg_logs[0], density=True, bins='auto', range=(-14.0, -4.0))[1]
        print(bins)
        neg_hists = [np.histogram(k, density=True, bins=bins)[0] for k in neg_logs]
        pos_hists = [np.histogram(k, density=True, bins=bins)[0] for k in pos_logs]
        # plt.hist(neg_hists, histtype='step', stacked=True, color=colors)
        # plt.(pos_hists, histtype='step', stacked=True, color=colors)
        ax = plt.subplot()

        neg_colors = cm.Reds(np.linspace(0.5, 1, 10))
        pos_colors = cm.Blues(np.linspace(0.5, 1, 10))
        cmap = plt.cm.coolwarm
        custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                        Line2D([0], [0], color=cmap(1.), lw=4)]
        # fig, ax = plt.subplots()
        for neg_hist, nc, pos_hist, pc in zip(neg_hists, neg_colors, pos_hists, pos_colors):
            ax.plot(range(len(neg_hists[0])), neg_hist, color=nc)
            ax.plot(range(len(pos_hists[0])), pos_hist, color=pc)
        ax.legend(custom_lines, ['CMV+, ' + pep, 'CMV-, ' + pep])
        ax.set_xticks([k for k in range(len(pos_hists[0])) if k % 5 == 0])
        ax.set_xticklabels([int(b) for i, b in enumerate(bins[:-1]) if i % 5 == 0])
        if i == 0:
            plt.xlabel('log(1 - x) normalized histograms for x > 0.98 scores')
            plt.title("Highest bin plotted histograms based on ERGO-" + args.model_type.upper() + " CMV peptide scores")
    plt.show()


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

    # read_predictions_from_file('ergo_predictions/LSTM/CMV+/HIP00594_NLVPMVATV.pickle')
    # exit()
    # save_predictions()
    # plot_score_histograms()
    # save_predictions_to_file('emerson_tcrs_with_temps/HIP00594.tsv', 'NLVPMVATV')
    # read_predictions_from_file('ergo_predictions2/LSTM/HIP00594_NLVPMVATV.pickle')
    # plot_multiple_peps_hists()
    # reg_score_hist()
    # score_hists_with_templates()
    # plot_single_hist()

# configurations:
# lstm cuda:1 --model_file=lstm_mcpas_specific__model.pt
# ae cuda:1 --model_file=ae_mcpas_specific__model.pt
