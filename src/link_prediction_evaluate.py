import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, auc
import torch.nn.functional as F
from info_nce import InfoNCE
# from src.logreg import LogReg
def load_training_data(f_name):
    edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            # words = line[:-1].split('\t')
            words = line[:-1].split()
            # print(words)
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_edges.append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    all_edges = list(set(all_edges))
    edge_data_by_type['Base'] = all_edges
    print('total training nodes: ' + str(len(all_nodes)))
    # print('Finish loading training data')
    return edge_data_by_type


def load_testing_data(f_name):
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            # words = line[:-1].split('\t')
            words = line[:-1].split()
            x, y = words[1], words[2]
            if int(words[3]) == 1:
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    # print('Finish loading testing data')
    return true_edge_data_by_type, false_edge_data_by_type


def get_score(local_model, node1, node2):
    """
    Calculate embedding similarity
    """
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        if type(vector1) != np.ndarray:
            vector1 = vector1.toarray()[0]
            vector2 = vector2.toarray()[0]

        return np.dot(vector1, vector2)
        # return np.dot(vector1, vector2) / ((np.linalg.norm(vector1) * np.linalg.norm(vector2) + 0.00000000000000001))
    except Exception as e:
        pass


def link_prediction_evaluate(model, true_edges, false_edges):
    """
    Link prediction process
    """

    true_list = list()
    prediction_list = list()
    true_num = 0

    # Calculate the similarity score of positive sample embedding
    for edge in true_edges:
        # tmp_score = get_score(model, str(edge[0]), str(edge[1])) # for amazon
        tmp_score = get_score(model, str(int(edge[0])), str(int(edge[1])))
        # tmp_score = get_score(model, str(int(edge[0] -1)), str(int(edge[1]-1)))
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

    # Calculate the the similarity score of negative sample embedding
    for edge in false_edges:
        # tmp_score = get_score(model, str(edge[0]), str(edge[1])) # for amazon
        tmp_score = get_score(model, str(int(edge[0])), str(int(edge[1])))
        # tmp_score = get_score(model, str(int(edge[0] -1)), str(int(edge[1]-1)))
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    # Determine the positive and negative sample threshold
    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    # Compare the similarity score with the threshold to predict whether the connection exists
    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)


def predict_model(model, file_name, feature, A,encode, eval_type, node_matching):
    """
    Link prediction training proces
    """

    training_data_by_type = load_training_data(file_name + '/train.txt')
    train_true_data_by_edge, train_false_data_by_edge = load_testing_data(file_name + '/train.txt')
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(file_name + '/valid.txt')
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name + '/test.txt')

    network_data = training_data_by_type
    edge_types = list(network_data.keys())  # ['1', '2', '3', '4', 'Base']
    edge_type_count = len(edge_types) - 1
    # edge_type_count = len(eval_type) - 1s

    device = torch.device('cpu')
    contrast = InfoNCE()
    aucs, f1s, prs = [], [], []
    validaucs, validf1s, validprs = [], [], []
    for _ in range(1):
        for iter_ in range(100):
            model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
            emb,near_embeds,far_embeds = model(feature, A,encode)

            emb_true_first = []
            emb_true_second = []
            emb_false_first = []
            emb_false_second = []

            for i in range(edge_type_count):
                if eval_type == 'all' or edge_types[i] in eval_type.split(','):
                    true_edges = train_true_data_by_edge[edge_types[i]]
                    false_edges = train_false_data_by_edge[edge_types[i]]

                for edge in true_edges:
                    # tmp_score = get_score(final_model, str(edge[0]), str(edge[1])) # for amazon
                    emb_true_first.append(emb[int(edge[0])])
                    emb_true_second.append(emb[int(edge[1])])

                for edge in false_edges:
                    # tmp_score = get_score(final_model, str(edge[0]), str(edge[1])) # for amazon
                    emb_false_first.append(emb[int(edge[0])])
                    emb_false_second.append(emb[int(edge[1])])

            emb_true_first = torch.cat(emb_true_first).reshape(-1, 200)
            emb_true_second = torch.cat(emb_true_second).reshape(-1, 200)
            emb_false_first = torch.cat(emb_false_first).reshape(-1, 200)
            emb_false_second = torch.cat(emb_false_second ).reshape(-1, 200)

            T1 = emb_true_first @ emb_true_second.T
            T2 = -(emb_false_first @ emb_false_second.T)

            pos_out = torch.diag(T1)
            neg_out = torch.diag(T2)
            # loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
            loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))+ 0.0001 * contrast(near_embeds, far_embeds)
            loss = loss.requires_grad_()

            opt.zero_grad()
            loss.backward()
            opt.step()

            td,near_embeds,far_embeds = model(feature, A,encode)
            td=td.detach().numpy()
            final_model = {}
            try:
                if node_matching == True:
                    for i in range(0, len(td)):
                        final_model[str(int(td[i][0]))] = td[i][1:]
                else:
                    for i in range(0, len(td)):
                        final_model[str(i)] = td[i]
            except:
                td = td.tocsr()
                if node_matching == True:
                    for i in range(0, td.shape[0]):
                        final_model[str(int(td[i][0]))] = td[i][1:]
                else:
                    for i in range(0, td.shape[0]):
                        final_model[str(i)] = td[i]
            train_aucs, train_f1s, train_prs = [], [], []
            valid_aucs, valid_f1s, valid_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            for i in range(edge_type_count):
                if eval_type == 'all' or edge_types[i] in eval_type.split(','):
                    train_auc, train_f1, train_pr = link_prediction_evaluate(final_model,
                                                                              train_true_data_by_edge[edge_types[i]],
                                                                              train_false_data_by_edge[edge_types[i]])
                    train_aucs.append(train_auc)
                    train_f1s.append(train_f1)
                    train_prs.append(train_pr)


                    valid_auc, valid_f1, valid_pr = link_prediction_evaluate(final_model,
                                                                              valid_true_data_by_edge[edge_types[i]],
                                                                              valid_false_data_by_edge[edge_types[i]])
                    valid_aucs.append(valid_auc)
                    valid_f1s.append(valid_f1)
                    valid_prs.append(valid_pr)

                    test_auc, test_f1, test_pr = link_prediction_evaluate(final_model,
                                                                          testing_true_data_by_edge[edge_types[i]],
                                                                          testing_false_data_by_edge[edge_types[i]])
                    test_aucs.append(test_auc)
                    test_f1s.append(test_f1)
                    test_prs.append(test_pr)

            print("{}\t{:.4f}\tweight_b:{}".format(iter_ + 1, loss.item(), model.weight_b))
            print("train_auc:{:.4f}\ttrain_pr:{:.4f}".format(np.mean(train_aucs),
                                                                              np.mean(train_prs)))
            print("valid_auc:{:.4f}\t\tvalid_pr:{:.4f}".format(np.mean(valid_aucs),
                                                                              np.mean(valid_prs)))
            print("test_auc:{:.4f}\ttest_pr:{:.4f}".format(np.mean(test_aucs),
                                                                           np.mean(test_prs)))
            validaucs.append(np.mean(valid_aucs))
            validf1s.append(np.mean(valid_f1s))
            validprs.append(np.mean(valid_prs))


            aucs.append(np.mean(test_aucs))
            f1s.append(np.mean(test_f1s))
            prs.append(np.mean(test_prs))

    max_iter_aucs = validaucs.index(max(validaucs))
    max_iter_f1s = validf1s.index(max(validf1s))
    max_iter_prs = validprs.index(max(validprs))

    return aucs[max_iter_aucs],  prs[max_iter_prs]
