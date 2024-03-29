import argparse
from data_loader import *
from ERGO_models import DoubleLSTMClassifier, AutoencoderLSTMClassifier
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np
import pickle
from sklearn import decomposition
from sklearn.metrics import roc_auc_score, roc_curve


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


def save_predictions_to_file(repertoire_filename, peptide, pred_dir):
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
    with open('/'.join([pred_dir, args.model_type.upper(),
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


def save_predictions(data_dir, pred_dir):
    cmv_peps = ['NLVPMVATV', 'VTEHDTLLY', 'TPRVTGGGAM']
    more_cmv_peps = ['YSEHPTFTSQY', 'QYDPVAALF', 'QIKVRVKMV']
    diabetes_peps = ['VLFGLGFAI', 'NFIRMVISNPAAT']
    peps = more_cmv_peps
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".tsv"):
                print(filepath)
                for pep in peps:
                    save_predictions_to_file(filepath, pep, pred_dir)


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


def get_repertoires_from_hla(hla):
    hla_csv = 'hla_types.csv'
    reps = []
    with open(hla_csv, 'r') as file:
        for row in file:
            row = row.strip().split(', ')
            if hla in row:
                reps.append(row[0])
    return reps


def get_rep_cmv_status(rep_id):
    with open('cmv_status', 'r') as file:
        for line in file:
            rep, status = line.strip().split(', ')
            if rep == rep_id:
                return status


def get_hlas(rep_id):
    with open('hla_types.csv', 'r') as file:
        for line in file:
            line = line.strip().split(', ')
            if line[0] == rep_id and len(line) > 0:
                return line[1:]


def plot_hists_matching_hla():
    cmv_peps = ['NLVPMVATV', 'VTEHDTLLY', 'TPRVTGGGAM']
    peps_hla = ['HLA-A*02', 'HLA-A*01', 'HLA-B*07']
    for i, pep in enumerate(cmv_peps):
        labels = []
        neg_p = []
        pos_p = []
        rep_ids = get_repertoires_from_hla(peps_hla[i])
        for rep_id in rep_ids:
            filepath = os.sep.join(['ergo_predictions2', args.model_type.upper(),
                                    rep_id + '_' + pep + '.pickle'])
            label = get_rep_cmv_status(rep_id)
            labels.append(label)
            if peps_hla[i] in get_hlas(rep_id):
                tcrs, temps, preds = read_predictions_from_file(filepath)
                if label == 'CMV-':
                    neg_p.append([pred for pred in preds if pred > 0.98])
                elif label == 'CMV+':
                    pos_p.append([pred for pred in preds if pred > 0.98])
        neg_logs = [np.log(1 - np.array(neg_bin)) for neg_bin in neg_p]
        pos_logs = [np.log(1 - np.array(pos_bin)) for pos_bin in pos_p]
        bins = np.histogram(neg_logs[0], density=True, bins='auto', range=(-14.0, -4.0))[1]
        # print(bins)
        neg_hists = [np.histogram(k, density=True, bins=bins)[0] for k in neg_logs]
        pos_hists = [np.histogram(k, density=True, bins=bins)[0] for k in pos_logs]
        ax = plt.subplot(1, 3, i+1)
        neg_colors = cm.Reds(np.linspace(0, 1, len(neg_hists)))
        pos_colors = cm.Blues(np.linspace(0.5, 1, len(pos_hists)))
        cmap = plt.cm.coolwarm
        custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                        Line2D([0], [0], color=cmap(1.), lw=4)]
        # fig, ax = plt.subplots()
        for neg_hist, nc, pos_hist, pc in zip(neg_hists, neg_colors, pos_hists, pos_colors):
            ax.plot(range(len(neg_hists[0])), neg_hist, color=nc)
            ax.plot(range(len(pos_hists[0])), pos_hist, color=pc)
        ax.legend(custom_lines, ['CMV+, ' + pep + ', ' + peps_hla[i], 'CMV-, ' + pep + ', ' + peps_hla[i]])
        ax.set_xticks([k for k in range(len(pos_hists[0])) if k % 5 == 0])
        ax.set_xticklabels([int(b) for i, b in enumerate(bins[:-1]) if i % 5 == 0])
        if i == 1:
            plt.xlabel('log(1 - x) normalized histograms for x > 0.98 scores')
            plt.title("Highest bin plotted histograms based on ERGO-" +args.model_type.upper() + " CMV peptide scores")
    plt.show()
    pass


def save_freq_peps_distribution(filepath):
    # save predictions fro more frequent peptides (not CMV)
    freq_peps = ['LPRRSGAAGA', 'GILGFVFTL', 'GLCTLVAML', 'SSYRRPVGI', 'RFYKTLRAEQASQ',
                 'SSLENFRAYV', 'ASNENMETM', 'ELAGIGILTV', 'LLWNGPMAV', 'HGIRNASFI']
    if filepath.endswith(".tsv"):
        print(filepath)
        for pep in freq_peps:
            save_predictions_to_file(filepath, pep)
    pass


def read_freq_peps_distribution(rep_id):
    freq_peps = ['LPRRSGAAGA', 'GILGFVFTL', 'GLCTLVAML', 'SSYRRPVGI', 'RFYKTLRAEQASQ',
                 'SSLENFRAYV', 'ASNENMETM', 'ELAGIGILTV', 'LLWNGPMAV', 'HGIRNASFI']
    cmv_peps = ['NLVPMVATV', 'VTEHDTLLY', 'TPRVTGGGAM']
    peps = cmv_peps + freq_peps
    # rep_id = filepath.split(os.sep)[-1][:-(1 + len(pep) + len('.pickle'))]
    tcr_preds = {}
    for pep in peps:
        file = os.sep.join(['ergo_predictions2', args.model_type.upper(),
                                rep_id + '_' + pep + '.pickle'])
        tcrs, temps, scores = read_predictions_from_file(file)
        for tcr, score in zip(tcrs, scores):
            try:
                tcr_preds[tcr].append(score)
            except KeyError:
                tcr_preds[tcr] = [score]
    # print(tcr_preds)
    index, k = 0, 0
    for tcr in tcr_preds:
        k += 1
        # if sum(tcr_preds[tcr]) < 1:
        if all(t < 0.98 for t in tcr_preds[tcr]):
            index += 1
            print(tcr)
    print(index, k)
    pass


def scores_pca():
    freq_peps = ['LPRRSGAAGA', 'GILGFVFTL', 'GLCTLVAML', 'SSYRRPVGI', 'RFYKTLRAEQASQ',
                 'SSLENFRAYV', 'ASNENMETM', 'ELAGIGILTV', 'LLWNGPMAV', 'HGIRNASFI']
    cmv_peps = ['NLVPMVATV', 'VTEHDTLLY', 'TPRVTGGGAM']
    '''
    for every peptide
        for every repertoire
            take peptide scores of 50000 samples of repertoire
            put as a column in a matrix
        apply PCA on the matrix rows
        reduce row dimension
    '''
    peps = cmv_peps #+ freq_peps
    num_samples = 5000
    rep_ids = set()
    for subdir, dirs, files in os.walk('ergo_predictions2'):
        for file in files:
            filepath = subdir + os.sep + file
            # correct_hla = pass
            if filepath.endswith(".pickle") and args.model_type in filepath.lower():
                rep_id = filepath.split(os.sep)[-1].split('_')[0]
                cmv_status = get_rep_cmv_status(rep_id)
                if not cmv_status is None:
                    rep_ids.add((rep_id, cmv_status))
                    if rep_id == 'HIP___':
                        print(cmv_status)
    rep_ids.remove(('HIP05763', 'CMV-'))
    rep_ids = list(rep_ids)[:200]
    for pep in peps:
        pep_scores = np.zeros((num_samples, len(rep_ids)))
        for j, rep_id in enumerate(rep_ids):
            file = os.sep.join(['ergo_predictions2', args.model_type.upper(),
                                rep_id[0] + '_' + pep + '.pickle'])
            tcrs, temps, scores = read_predictions_from_file(file)
            if len(scores) >= num_samples:
                pep_scores[:, j] = scores[:num_samples]
            else:
                print(rep_id[0])
        pep_scores = np.transpose(pep_scores)
        pca = decomposition.PCA(n_components=2)
        pca.fit(pep_scores)
        pep_scores = pca.transform(pep_scores)
        status_to_color = {'CMV+': 'royalblue', 'CMV-': 'tomato'}
        colors = [status_to_color[rep_id[1]] for rep_id in rep_ids]
        custom_lines = [Line2D([0], [0], color='royalblue', lw=4),
                        Line2D([0], [0], color='tomato', lw=4)]
        # plt.clf()
        fig, ax = plt.subplots()
        ax.scatter(pep_scores[:, 0], pep_scores[:, 1], color=colors)
        ax.legend(custom_lines, ['CMV+', 'CMV-'])
        ax.set_title('Repertoires Scores PCA, peptide: ' + pep)
        plt.savefig('pca_scores_' + pep)
        # plt.show()
    # print(pep_scores.shape)
    pass


def histograms_with_reads():
    # read prediction files from directory
    # take only > 0.98
    # multiply score with number of reads
    # get CMV status from file (and from function)
    # plot histograms
    cmv_peps = ['NLVPMVATV', 'VTEHDTLLY', 'TPRVTGGGAM']
    for i, pep in enumerate(cmv_peps):
        labels = []
        neg_p = []
        pos_p = []
        index = 0
        for subdir, dirs, files in os.walk('ergo_predictions_reads'):
            for file in files:
                index += 1
                if index > 300:
                    break
                filepath = subdir + os.sep + file
                if filepath.endswith(pep + ".pickle") and args.model_type in filepath.lower():
                    print(filepath)
                    rep_id = filepath.split(os.sep)[-1].split('_')[0]
                    label = get_rep_cmv_status(rep_id)
                    print(label)
                    labels.append(label)
                    tcrs, reads, preds = read_predictions_from_file(filepath)
                    product = [[k] * int(c) for k, c in zip(preds, reads) if c != 'null']
                    flat = []
                    for l in product:
                        flat.extend(l)
                    preds = flat
                    if label == 'CMV-':
                        neg_p.append([pred for pred in preds if pred > 0.98])
                    elif label == 'CMV+':
                        pos_p.append([pred for pred in preds if pred > 0.98])
        neg_logs = [np.log(1 - np.array(neg_bin)) for neg_bin in neg_p]
        pos_logs = [np.log(1 - np.array(pos_bin)) for pos_bin in pos_p]
        bins = np.histogram(neg_logs[0], density=True, bins='auto', range=(-14.0, -4.0))[1]
        neg_hists = [np.histogram(k, density=True, bins=bins)[0] for k in neg_logs]
        pos_hists = [np.histogram(k, density=True, bins=bins)[0] for k in pos_logs]
        ax = plt.subplot(1, 3, i + 1)
        neg_colors = cm.Reds(np.linspace(0.5, 1, len(neg_logs)))
        pos_colors = cm.Blues(np.linspace(0.5, 1, len(pos_logs)))
        cmap = plt.cm.coolwarm
        custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                        Line2D([0], [0], color=cmap(1.), lw=4)]
        for neg_hist, nc in zip(neg_hists, neg_colors):
            ax.plot(range(len(neg_hists[0])), neg_hist, color=nc)
        for pos_hist, pc in zip(pos_hists, pos_colors):
            ax.plot(range(len(pos_hists[0])), pos_hist, color=pc)
        ax.legend(custom_lines, ['CMV+, ' + pep, 'CMV-, ' + pep])
        ax.set_xticks([k for k in range(len(pos_hists[0])) if k % 20 == 0])
        ax.set_xticklabels([int(b) for i, b in enumerate(bins[:-1]) if i % 20 == 0])
        if i == 1:
            plt.xlabel('log(1 - x) normalized histograms for x > 0.98 scores')
            plt.title("Highest bin plotted histograms based on ERGO-" + args.model_type.upper() + " CMV peptide scores")
    plt.show()
    #
    pass


def cumulative_with_reads():
    cmv_peps = ['NLVPMVATV', 'VTEHDTLLY', 'TPRVTGGGAM']
    for i, pep in enumerate(cmv_peps):
        labels = []
        neg_p = []
        pos_p = []
        index = 0
        for subdir, dirs, files in os.walk('ergo_predictions_reads'):
            for file in files:
                index += 1
                if index > 100:
                    break
                filepath = subdir + os.sep + file
                if filepath.endswith(pep + ".pickle") and args.model_type in filepath.lower():
                    print(filepath)
                    rep_id = filepath.split(os.sep)[-1].split('_')[0]
                    label = get_rep_cmv_status(rep_id)
                    print(label)
                    labels.append(label)
                    tcrs, reads, preds = read_predictions_from_file(filepath)
                    product = [[k] * int(c) for k, c in zip(preds, reads) if c != 'null']
                    flat = []
                    for l in product:
                        flat.extend(l)
                    preds = flat
                    if label == 'CMV-':
                        neg_p.append([pred for pred in preds if pred > 0.98])
                    elif label == 'CMV+':
                        pos_p.append([pred for pred in preds if pred > 0.98])
        neg_logs = [np.log(1 - np.array(neg_bin)) for neg_bin in neg_p]
        pos_logs = [np.log(1 - np.array(pos_bin)) for pos_bin in pos_p]
        bins = np.histogram(neg_logs[0], density=True, bins='auto', range=(-14.0, -4.0))[1]
        # neg_hists = [np.histogram(k, density=True, bins=bins)[0] for k in neg_logs]
        # pos_hists = [np.histogram(k, density=True, bins=bins)[0] for k in pos_logs]
        ax = plt.subplot(1, 3, i + 1)
        neg_colors = cm.Reds(np.linspace(0.5, 1, len(neg_logs)))
        pos_colors = cm.Blues(np.linspace(0.5, 1, len(pos_logs)))
        cmap = plt.cm.coolwarm
        custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                        Line2D([0], [0], color=cmap(1.), lw=4)]
        print(neg_logs)
        print(pos_logs)
        print(len(neg_logs))
        #for neg_log in neg_logs:
        #    neg_log /= len(neg_logs)
        #for pos_log in pos_logs:
        #    pos_log /= len(pos_logs)
        print(neg_logs)
        print(pos_logs)
        print(len(neg_logs), len(pos_logs))
        m = min(len(pos_logs), len(neg_logs))
        ax.hist(neg_logs[:m], color=neg_colors[:m], cumulative=True, bins=bins)
        ax.hist(pos_logs[:m], color=pos_colors[:m], cumulative=True, bins=bins)

        #for neg_hist, nc in zip(neg_hists, neg_colors):
        #    ax.hist(neg_hist, color=nc)
        #for pos_hist, pc in zip(pos_hists, pos_colors):
        #    ax.hist(pos_hist, color=pc)
        ax.legend(custom_lines, ['CMV+, ' + pep, 'CMV-, ' + pep])
        #ax.set_xticks([k for k in range(len(pos_hists[0])) if k % 20 == 0])
        #ax.set_xticklabels([int(b) for i, b in enumerate(bins[:-1]) if i % 20 == 0])
        if i == 1:
            plt.xlabel('log(1 - x) normalized histograms for x > 0.98 scores')
            plt.title("Highest bin plotted histograms based on ERGO-" + args.model_type.upper() + " CMV peptide scores")
    plt.show()
    pass


def hists_reads_cutoff():
    # read prediction files from directory
    # take only > 0.98
    # take only reads > 10
    # get CMV status from file (and from function)
    # plot histograms
    cmv_peps = ['NLVPMVATV', 'VTEHDTLLY', 'TPRVTGGGAM']
    for i, pep in enumerate(cmv_peps):
        labels = []
        neg_p = []
        pos_p = []
        index = 0
        for subdir, dirs, files in os.walk('ergo_predictions_reads'):
            for file in files:
                index += 1
                if index > 300:
                    break
                filepath = subdir + os.sep + file
                if filepath.endswith(pep + ".pickle") and args.model_type in filepath.lower():
                    print(filepath)
                    rep_id = filepath.split(os.sep)[-1].split('_')[0]
                    label = get_rep_cmv_status(rep_id)
                    print(label)
                    labels.append(label)
                    tcrs, reads, preds = read_predictions_from_file(filepath)
                    if label == 'CMV-':
                        neg_p.append([pred for pred in preds if pred > 0.98])
                    elif label == 'CMV+':
                        pos_p.append([pred for pred in preds if pred > 0.98])
        neg_logs = [np.log(1 - np.array(neg_bin)) for neg_bin in neg_p]
        pos_logs = [np.log(1 - np.array(pos_bin)) for pos_bin in pos_p]
        bins = np.histogram(neg_logs[0], density=True, bins='auto', range=(-14.0, -4.0))[1]
        neg_hists = [np.histogram(k, density=True, bins=bins)[0] for k in neg_logs]
        pos_hists = [np.histogram(k, density=True, bins=bins)[0] for k in pos_logs]
        ax = plt.subplot(1, 3, i + 1)
        neg_colors = cm.Reds(np.linspace(0.5, 1, len(neg_logs)))
        pos_colors = cm.Blues(np.linspace(0.5, 1, len(pos_logs)))
        cmap = plt.cm.coolwarm
        custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                        Line2D([0], [0], color=cmap(1.), lw=4)]
        for neg_hist, nc in zip(neg_hists, neg_colors):
            ax.plot(range(len(neg_hists[0])), neg_hist, color=nc)
        for pos_hist, pc in zip(pos_hists, pos_colors):
            ax.plot(range(len(pos_hists[0])), pos_hist, color=pc)
        ax.legend(custom_lines, ['CMV+, ' + pep, 'CMV-, ' + pep])
        ax.set_xticks([k for k in range(len(pos_hists[0])) if k % 20 == 0])
        ax.set_xticklabels([int(b) for i, b in enumerate(bins[:-1]) if i % 20 == 0])
        if i == 1:
            plt.xlabel('log(1 - x) normalized histograms for x > 0.98 scores')
            plt.title("Highest bin plotted histograms based on ERGO-" + args.model_type.upper() + " CMV peptide scores")
    plt.show()


def accumulating_score_distribution():
    cmv_peps = ['NLVPMVATV', 'VTEHDTLLY', 'TPRVTGGGAM']
    for i, pep in enumerate(cmv_peps):
        labels = []
        neg_p = []
        pos_p = []
        index = 0
        for subdir, dirs, files in os.walk('ergo_predictions_reads'):
            for file in files:
                index += 1
                if index > 300:
                    break
                filepath = subdir + os.sep + file
                if filepath.endswith(pep + ".pickle") and args.model_type in filepath.lower():
                    print(filepath)
                    rep_id = filepath.split(os.sep)[-1].split('_')[0]
                    label = get_rep_cmv_status(rep_id)
                    print(label)
                    labels.append(label)
                    tcrs, reads, preds = read_predictions_from_file(filepath)
                    product = [[k] * int(np.sqrt(int(c))) for k, c in zip(preds, reads) if c != 'null']
                    flat = []
                    for l in product:
                        flat.extend(l)
                    preds = flat
                    if label == 'CMV-':
                        neg_p.append([pred for pred in preds if pred > 0.98])
                    elif label == 'CMV+':
                        pos_p.append([pred for pred in preds if pred > 0.98])
        neg_logs = [np.log(1 - np.array(neg_bin)) for neg_bin in neg_p]
        pos_logs = [np.log(1 - np.array(pos_bin)) for pos_bin in pos_p]
        bins = np.histogram(neg_logs[0], density=True, bins='auto', range=(-14.0, -4.0))[1]
        # neg_hists = [np.histogram(k, density=True, bins=bins)[0] for k in neg_logs]
        # pos_hists = [np.histogram(k, density=True, bins=bins)[0] for k in pos_logs]
        ax = plt.subplot(1, 3, i + 1)
        neg_colors = cm.Reds(np.linspace(0.5, 1, len(neg_logs)))
        pos_colors = cm.Blues(np.linspace(0.5, 1, len(pos_logs)))
        cmap = plt.cm.coolwarm
        custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                        Line2D([0], [0], color=cmap(1.), lw=4)]
        print(neg_logs)
        print(pos_logs)
        print(len(neg_logs))
        # for neg_log in neg_logs:
        #    neg_log /= len(neg_logs)
        # for pos_log in pos_logs:
        #    pos_log /= len(pos_logs)
        print(neg_logs)
        print(pos_logs)
        print(len(neg_logs), len(pos_logs))
        m = min(len(pos_logs), len(neg_logs))
        ax.hist(neg_logs[:m], color=neg_colors[:m], cumulative=True, bins=bins, histtype='step', density=True)
        ax.hist(pos_logs[:m], color=pos_colors[:m], cumulative=True, bins=bins, histtype='step', density=True)

        # for neg_hist, nc in zip(neg_hists, neg_colors):
        #    ax.hist(neg_hist, color=nc)
        # for pos_hist, pc in zip(pos_hists, pos_colors):
        #    ax.hist(pos_hist, color=pc)
        ax.legend(custom_lines, ['CMV+, ' + pep, 'CMV-, ' + pep])
        # ax.set_xticks([k for k in range(len(pos_hists[0])) if k % 20 == 0])
        # ax.set_xticklabels([int(b) for i, b in enumerate(bins[:-1]) if i % 20 == 0])
        if i == 1:
            plt.xlabel('log(1 - x) * floor(sqrt(reads) normalized histograms for x > 0.98 scores')
            plt.title("Score cumulative histograms based on ERGO-" + args.model_type.upper() + " CMV peptide scores")
    plt.show()
    pass


def cumulative_hist_roc_curve():
    cmv_peps = ['NLVPMVATV', 'VTEHDTLLY', 'TPRVTGGGAM']
    for i, pep in enumerate(cmv_peps):
        for function, name in zip([lambda x: x, lambda x: np.log(x), lambda x: np.sqrt(x)],
                                  ['reads', 'floor(log(reads))', 'floor(sqrt(reads))']):
            labels = []
            neg_p = []
            pos_p = []
            index = 0
            for subdir, dirs, files in os.walk('ergo_predictions_reads'):
                for file in files:
                    index += 1
                    if index > 100:
                        break
                    filepath = subdir + os.sep + file
                    if filepath.endswith(pep + ".pickle") and args.model_type in filepath.lower():
                        print(filepath)
                        rep_id = filepath.split(os.sep)[-1].split('_')[0]
                        label = get_rep_cmv_status(rep_id)
                        print(label)
                        labels.append(label)
                        tcrs, reads, preds = read_predictions_from_file(filepath)
                        product = [[k] * int(function(int(c))) for k, c in zip(preds, reads) if c != 'null']
                        flat = []
                        for l in product:
                            flat.extend(l)
                        preds = flat
                        if label == 'CMV-':
                            neg_p.append([pred for pred in preds if pred > 0.98])
                        elif label == 'CMV+':
                            pos_p.append([pred for pred in preds if pred > 0.98])
            neg_logs = [np.log(1 - np.array(neg_bin)) for neg_bin in neg_p]
            pos_logs = [np.log(1 - np.array(pos_bin)) for pos_bin in pos_p]
            bins = np.histogram(neg_logs[0], density=True, bins='auto', range=(-14.0, -4.0))[1]
            y_true = []
            y_scores = []
            areas = []
            plt.figure(1)
            for log in neg_logs:
                values, bins, _ = plt.hist(log, cumulative=True, bins=bins, density=True)
                area = sum(np.diff(bins) * values)
                y_true.append(0)
                y_scores.append(area)
                areas.append(area)
            for log in pos_logs:
                values, bins, _ = plt.hist(log, cumulative=True, bins=bins, density=True)
                area = sum(np.diff(bins) * values)
                y_true.append(1)
                y_scores.append(area)
                areas.append(area)
            avg_area = np.average(area)
            y_scores /= avg_area * 2  # make average score to be 0.5
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            # plt.cla()
            plt.figure(2)
            plt.title('ROC curves of cumulative scores histograms area')
            ax = plt.subplot(1, 3, i + 1)
            ax.plot(fpr, tpr, label=name + ', auc: ' + str(np.round(roc_auc_score(y_true, y_scores), decimals=2)))
            ax.legend()
            ax.set_xlabel(pep)
    plt.show()


def cumulative_hist_grid():
    # get cumulative hists.
    # set a grid of (x,y) points
    # plot on x the fraction of blue/red for which f(x) > y
    nx, ny = (20, 20)
    x = np.linspace(-14, -4, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    cmv_peps = ['NLVPMVATV', 'VTEHDTLLY', 'TPRVTGGGAM']
    for i, pep in enumerate(cmv_peps):
        for j, (function, name) in enumerate(zip([lambda x: x, lambda x: np.log(x), lambda x: np.sqrt(x)],
                                  ['reads', 'floor(log(reads))', 'floor(sqrt(reads))'])):
            labels = []
            neg_p = []
            pos_p = []
            index = 0
            for subdir, dirs, files in os.walk('ergo_predictions_reads'):
                for file in files:
                    if index < 100:
                        index += 1
                        continue
                    index += 1
                    # to
                    if index > 200:
                        break
                    filepath = subdir + os.sep + file
                    if filepath.endswith(pep + ".pickle") and args.model_type in filepath.lower():
                        print(filepath)
                        rep_id = filepath.split(os.sep)[-1].split('_')[0]
                        label = get_rep_cmv_status(rep_id)
                        print(label)
                        labels.append(label)
                        tcrs, reads, preds = read_predictions_from_file(filepath)
                        product = [[k] * int(function(int(c))) for k, c in zip(preds, reads) if c != 'null']
                        flat = []
                        for l in product:
                            flat.extend(l)
                        preds = flat
                        if label == 'CMV-':
                            neg_p.append([pred for pred in preds if pred > 0.98])
                        elif label == 'CMV+':
                            pos_p.append([pred for pred in preds if pred > 0.98])
            neg_logs = [np.log(1 - np.array(neg_bin)) for neg_bin in neg_p]
            pos_logs = [np.log(1 - np.array(pos_bin)) for pos_bin in pos_p]
            bins = np.histogram(neg_logs[0], density=True, bins='auto', range=(-14.0, -4.0))[1]
            y_true = []
            y_scores = []
            plt.figure(1)
            result_x = []
            result_y = []
            neg_hists = []
            pos_hists = []
            for log in neg_logs:
                values, _, _ = plt.hist(log, cumulative=True, bins=bins, density=True)
                neg_hists.append(values)
            for log in pos_logs:
                values, _, _ = plt.hist(log, cumulative=True, bins=bins, density=True)
                pos_hists.append(values)
            closest_bin = lambda x: min(enumerate(bins), key=lambda t: abs(t[1] - x))
            for x, y in zip(xv, yv):
                above_neg_count = 0
                for hist in neg_hists:
                    bin_index, bin = closest_bin(x)
                    if hist[max(0, bin_index - 1)] >= y:
                        above_neg_count += 1
                result_x.append(above_neg_count / len(neg_hists))
                above_pos_count = 0
                for hist in pos_hists:
                    bin_index, bin = closest_bin(x)
                    if hist[max(0, bin_index - 1)] >= y:
                        above_pos_count += 1
                result_y.append(above_pos_count / len(pos_hists))
            plt.clf()
            plt.figure(2)
            # plt.title('')
            ax = plt.subplot(3, 3, i * 3 + j + 1)
            ax.scatter(result_x, result_y)
            ax.set_title(name + ', ' + pep)
    plt.show()
    pass


# Diabetes
def get_diabetes_status(rep_id):
    with open('diabetes_status.csv') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            if line[0] == rep_id + '.tsv':
                return line[1]
    pass


def diabetes_cumulative_hists():
    diabetes_peps = ['VLFGLGFAI', 'NFIRMVISNPAAT']
    factor = lambda x: x
    for i, pep in enumerate(diabetes_peps):
        labels = []
        t1d_p = []
        ab_p = []
        control_p = []
        index = 0
        for subdir, dirs, files in os.walk('diabetes_predictions'):
            for file in files:
                '''
                index += 1
                if index > 300:
                    break
                '''
                filepath = subdir + os.sep + file
                if filepath.endswith(pep + ".pickle") and args.model_type in filepath.lower():
                    print(filepath)
                    rep_id = '_'.join(filepath.split(os.sep)[-1].split('_')[:-1])
                    print(rep_id)
                    label = get_diabetes_status(rep_id)
                    print(label)
                    labels.append(label)
                    tcrs, reads, preds = read_predictions_from_file(filepath)
                    product = [[k] * int(factor((int(c)))) for k, c in zip(preds, reads) if c != 'null']
                    flat = []
                    for l in product:
                        flat.extend(l)
                    preds = flat
                    if label == 'T1D':
                        t1d_p.append([pred for pred in preds if pred > 0.98])
                    elif label == 'AB+':
                        ab_p.append([pred for pred in preds if pred > 0.98])
                    elif label == 'Control':
                        control_p.append([pred for pred in preds if pred > 0.98])
        t1d_logs = [np.log(1 - np.array(bin)) for bin in t1d_p]
        ab_logs = [np.log(1 - np.array(bin)) for bin in ab_p]
        control_logs = [np.log(1 - np.array(bin)) for bin in control_p]
        bins = np.histogram(t1d_logs[0], density=True, bins='auto', range=(-14.0, -4.0))[1]
        ax = plt.subplot(1, 2, i + 1)
        t1d_colors = cm.Reds(np.linspace(0.5, 1, len(t1d_logs)))
        ab_colors = cm.Blues(np.linspace(0.5, 1, len(ab_logs)))
        control_colors = cm.Greens(np.linspace(0.5, 1, len(control_logs)))
        custom_lines = [Line2D([0], [0], color='red', lw=4),
                        Line2D([0], [0], color='blue', lw=4),
                        Line2D([0], [0], color='green', lw=4)]
        m = min(len(t1d_logs), len(control_logs), len(ab_logs))
        ax.hist(t1d_logs[:m], color=t1d_colors[:m], cumulative=True, bins=bins, histtype='step', density=True)
        ax.hist(ab_logs[:m], color=ab_colors[:m], cumulative=True, bins=bins, histtype='step', density=True)
        ax.hist(control_logs[:m], color=control_colors[:m], cumulative=True, bins=bins, histtype='step', density=True)
        ax.legend(custom_lines, ['T1D, ' + pep, 'AB+, ' + pep, 'Control, ' + pep])
        plt.suptitle("Score cumulative histograms based on ERGO-" + args.model_type.upper() + " Diabetes peptide scores")
        #if i == 1:
        #    plt.xlabel('log(1 - x) * floor(sqrt(reads) normalized histograms for x > 0.98 scores')
        #    plt.title("Score cumulative histograms based on ERGO-" + args.model_type.upper() + " CMV peptide scores")
    plt.show()
    pass


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
    # get_repertoires_from_hla('HLA-A*02')
    # plot_hists_matching_hla()
    # save_freq_peps_distribution('emerson_tcrs_with_temps/HIP00594.tsv')
    # read_freq_peps_distribution('HIP00594')
    # scores_pca()
    # histograms_with_reads()
    # cumulative_with_reads()
    # hists_reads_cutoff()
    # accumulating_score_distribution()
    # cumulative_hist_roc_curve()
    # cumulative_hist_grid()
    # save_predictions('diabetes_tcr_counts', 'diabetes_predictions')
    # save_predictions('emerson_tcrs_with_reads', 'ergo_predictions_reads')
    diabetes_cumulative_hists()

# configurations:
# lstm cuda:1 --model_file=lstm_mcpas_specific__model.pt
# ae cuda:1 --model_file=ae_mcpas_specific__model.pt
