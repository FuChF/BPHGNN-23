import numpy as np
import scipy.io as sio
import pickle as pkl
import torch.nn as nn
from sklearn.metrics import f1_score
import time
from info_nce import InfoNCE
import torch

from src.logreg import LogReg


def load_data(dataset, datasetfile_type):
    """"Get the label of node classification, training set, verification machine and test set"""
    if datasetfile_type == 'mat':
        data = sio.loadmat('data/{}.mat'.format(dataset))
    else:
        data = pkl.load(open('data/{}.pkl'.format(dataset), "rb"))
    try:
        labels = data['label']
    except:
        labels = data['labelmat']

    # idx_train = data['train_idx'].ravel()
    # try:
    #     idx_val = data['valid_idx'].ravel()
    # except:
    #     idx_val = data['val_idx'].ravel()
    # idx_test = data['test_idx'].ravel()

    # # # # imdb_small
    # t_v_t=sio.loadmat('data/imdb_small/imdb_small_80.mat')
    # idx_train = t_v_t['train_idx'].ravel()
    # idx_val = t_v_t['valid_idx'].ravel()
    # idx_test = t_v_t['test_idx'].ravel()


    # # # # dblp_small
    # t_v_t=sio.loadmat('data/dblp_small/dblp_small_80.mat')
    # idx_train = t_v_t['train_idx'].ravel()
    # idx_val = t_v_t['valid_idx'].ravel()
    # idx_test = t_v_t['test_idx'].ravel()

    # # # # alibaba
    # t_v_t=sio.loadmat('data/small_alibaba_1_10/small_alibaba_1_10_20.mat')
    # idx_train = t_v_t['train_idx'].ravel()
    # idx_val = t_v_t['valid_idx'].ravel()
    # idx_test = t_v_t['test_idx'].ravel()

    # # # alibaba_small
    t_v_t=sio.loadmat('data/alibaba_small/alibaba_small_20.mat')
    idx_train = t_v_t['train_idx'].ravel()
    idx_val = t_v_t['valid_idx'].ravel()
    idx_test = t_v_t['test_idx'].ravel()


    # # # # alibaba_large
    # t_v_t=sio.loadmat('data/alibaba_large/alibaba_large_20.mat')
    # idx_train = t_v_t['train_idx'].ravel()
    # idx_val = t_v_t['valid_idx'].ravel()
    # idx_test = t_v_t['test_idx'].ravel()


    # imdb
    # t_v_t=sio.loadmat('data/IMDB/imdb_60.mat')
    # idx_train = t_v_t['train_idx'].ravel()
    # idx_val = t_v_t['valid_idx'].ravel()
    # idx_test = t_v_t['test_idx'].ravel()

    return labels, idx_train.astype(np.int32) - 1, idx_val.astype(np.int32) - 1, idx_test.astype(np.int32) - 1


def node_classification_evaluate(model, feature, A,encode,new_adj, file_name, file_type, device, isTest=True):
    """Node classification training process"""

    embeds,near_embeds,far_embeds = model(feature, A,encode,new_adj)

    labels, idx_train, idx_val, idx_test = load_data(file_name, file_type)

    try:
        labels = labels.todense()
    except:
        pass
    labels = labels.astype(np.int16)
    embeds = torch.FloatTensor(embeds[np.newaxis]).to(device)
    # near_embs = torch.FloatTensor(near_embs[np.newaxis]).to(device)
    # far_embs = torch.FloatTensor(far_embs[np.newaxis]).to(device)

    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    hid_units = embeds.shape[2]
    nb_classes = labels.shape[2]
    xent = nn.CrossEntropyLoss()
    contrast = InfoNCE()
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]
    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []

    for _ in range(1):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.01}, {'params': log.parameters()}], lr=0.001, weight_decay=0.0005)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        starttime = time.time()
        for iter_ in range(200):
            embeds,near_embeds,far_embeds = model(feature, A,encode,new_adj)
            # print(embeds)
            embeds = torch.FloatTensor(embeds[np.newaxis]).to(device)
            train_embs = embeds[0, idx_train]
            val_embs = embeds[0, idx_val]
            test_embs = embeds[0, idx_test]


            # near_embs=torch.FloatTensor(near_embs[np.newaxis]).to(device)
            # far_embs = torch.FloatTensor(far_embs[np.newaxis]).to(device)
            #
            # train_near_embs= near_embs[0, idx_train]
            # train_far_embs = far_embs[0, idx_train]


            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)

            # loss = xent(logits, train_lbls)
            loss = xent(logits, train_lbls) + 0.01 * contrast(near_embeds, far_embeds)
            # loss = contrast(near_embeds, far_embeds)
            loss.backward()
            opt.step()




            logits_tra = log(train_embs)
            preds = torch.argmax(logits_tra, dim=1)

            tra_f1_macro = f1_score(train_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
            tra_f1_micro = f1_score(train_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')
            print("===============================train{}\t{:.4f}\t{:.4f}\t{:.4f}".format(iter_ + 1, loss.item(),
                                                                                          tra_f1_macro,
                                                                                          tra_f1_micro))




            logits_val = log(val_embs)
            preds = torch.argmax(logits_val, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')

            print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(iter_ + 1, loss.item(), val_acc, val_f1_macro,
                                                              val_f1_micro))
            print("weight_b:{}".format(model.weight_b))

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits_test = log(test_embs)
            preds = torch.argmax(logits_test, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')
            print("test_f1-ma: {:.4f}\ttest_f1-mi: {:.4f}".format(test_f1_macro, test_f1_micro))

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)

        endtime = time.time()

        print('time: {:.10f}'.format(endtime - starttime))

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])

        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])


    if isTest:
        print("\t[Classification] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(np.mean(macro_f1s),
                                                                                                np.std(macro_f1s),
                                                                                                np.mean(micro_f1s),
                                                                                                np.std(micro_f1s)))
    else:
        return np.mean(macro_f1s), np.mean(micro_f1s)

    return np.mean(macro_f1s), np.mean(micro_f1s)