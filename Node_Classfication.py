import torch
from scipy.io import loadmat
from scipy.sparse import csc_matrix
import numpy as np
from src.Utils import load_our_data, get_model
from src.args import get_citation_args
from src.node_classfication_evaluate import node_classification_evaluate

args = get_citation_args()

# IMDB
# args.dataset = 'imdb_1_10'
# eval_name = r'data/imdb_1_10'
# net_path = r"data/IMDB/imdb_1_10.mat"
# savepath = r'data/imdb_embedding_1_10'
# eval_name = r'imdb_1_10'
# encode_path=r'data/IMDB/imdb_encode_num.pt'
# adj_path=r'data/IMDB/imdb_new_edge.pt'

# DBLP
# args.dataset = 'DBLP'
# eval_name = r'data/DBLP'
# net_path = r"data/dblp/DBLP.mat"
# savepath = r'data/DBLP_embedding'
# eval_name = r'DBLP'

# Aminer
# args.dataset = 'Aminer_10k_4class'
# eval_name = r'Aminer_10k_4class'
# net_path = r'../data/Aminer_1_13/Aminer_10k_4class.mat'
# savepath = r'embedding/Aminer_10k_4class_aminer_embedding_'
# eval_name = r'Aminer_10k_4class'

#alibaba_small
args.dataset = 'alibaba_small'
eval_name = r'alibaba_small'
net_path = r'data/alibaba_small/alibaba_small.mat'
savepath = r'data/alibaba_small'
eval_name = r'alibaba_small'
encode_path=r'data/alibaba_small/alibaba_small_encode.pt'
# adj_path=r'data/dblp_small/dblp_small_new_edge.pt'
# graph_path = r'data/small_alibaba_1_10/alibaba_multi.mat'



# #alibaba_large
# args.dataset = 'alibaba_large'
# eval_name = r'alibaba_large'
# net_path = r'data/alibaba_large/alibaba_large.mat'
# savepath = r'data/alibaba_large'
# eval_name = r'alibaba_large'
# encode_path=r'data/alibaba_large/alibaba_large_encode.pt'
# adj_path=r'data/dblp_small/dblp_small_new_edge.pt'
# graph_path = r'data/small_alibaba_1_10/alibaba_multi.mat'


# alibaba
# args.dataset = 'small_alibaba_1_10'
# eval_name = r'small_alibaba_1_10'
# net_path = r'data/small_alibaba_1_10/small_alibaba_1_10.mat'
# savepath = r'data/alibaba_embedding_'
# eval_name = r'small_alibaba_1_10'
# encode_path=r'data/small_alibaba_1_10/alibaba_encode.pt'
# adj_path=r'data/small_alibaba_1_10/smallalibaba_new_edge.pt'
# graph_path = r'data/small_alibaba_1_10/alibaba_multi.mat'

# #dblp_small
# args.dataset = 'dblp_small'
# eval_name = r'dblp_small'
# net_path = r'data/dblp_small/dblp_small.mat'
# savepath = r'data/dblp_small'
# eval_name = r'dblp_small'
# encode_path=r'data/dblp_small/dblp_small_encode.pt'
# adj_path=r'data/dblp_small/dblp_small_new_edge.pt'
# graph_path = r'data/small_alibaba_1_10/alibaba_multi.mat'

# #imdb_small
# args.dataset = 'imdb_small'
# eval_name = r'imdb_small'
# net_path = r'data/imdb_small/imdb_small.mat'
# savepath = r'data/imdb_small'
# eval_name = r'imdb_small'
# encode_path=r'data/imdb_small/imdb_small_encode.pt'
# adj_path=r'data/dblp_small/dblp_small_new_edge.pt'
# graph_path = r'data/small_alibaba_1_10/alibaba_multi.mat'


print('start')
# new_adj = torch.load(adj_path)
mat = loadmat(net_path)
encode=torch.load(encode_path)
print('end')
try:
    train = mat['A']
except:
    try:
        train = mat['train']+mat['valid']+mat['test']
    except:
        try:
            train = mat['train_full']+mat['valid_full']+mat['test_full']
        except:
            try:
                train = mat['edges']
            except:
                train = mat['edge']

try:
    feature = mat['full_feature']
except:
    try:
        feature = mat['feature']
    except:
        try:
            feature = mat['features']
        except:
            feature = mat['node_feature']
A = train[0][0]
feature = csc_matrix(feature) if type(feature) != csc_matrix else feature

if net_path == 'imdb_1_10.mat':
    A = train[0]
elif args.dataset == 'Aminer_10k_4class':
    A = [[mat['PAP'], mat['PCP'], mat['PTP'] ]]
    feature = mat['node_feature']
    feature = csc_matrix(feature) if type(feature) != csc_matrix else feature
else:
    A = train
if args.dataset == 'small_alibaba_1_10':
    mat=loadmat(graph_path)
    A=mat['edge']
if args.dataset == 'dblp_small':
    mat=loadmat(net_path)
    A=mat['edge']
if args.dataset == 'imdb_small':
    mat=loadmat(net_path)
    A=mat['edge']
if args.dataset == 'alibaba_large':
    mat=loadmat(net_path)
    A=mat['edge']
if args.dataset == 'alibaba_small':
    mat=loadmat(net_path)
    A=mat['edge']


def normarlize(H):
    # DV = np.sum(H, axis=1)
    # DV += 1e-12
    # DV2 = np.mat(np.diag(np.power(DV, -1)))
    # G = DV2 * H

    DV = torch.sum(H, dim=1)
    DV += 1e-12
    DV2 = torch.diag(torch.pow(DV, -1))
    G = torch.mm(DV2,H)

    return G

# new_adj=normarlize(new_adj)



adj, features, labels, idx_train, idx_val, idx_test = load_our_data(args.dataset, args.cuda)

model = get_model(args.model, features.size(1), labels.max().item()+1, A, args.hidden, args.out, args.dropout, False)

f1_ma, f1_mi = node_classification_evaluate(model, feature, A,encode, eval_name, file_type='mat', device=torch.device('cpu'))

print('Test F1-ma: {:.10f}, F1-mi: {:.10f}'.format(f1_ma, f1_mi))
