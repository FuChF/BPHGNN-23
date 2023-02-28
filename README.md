# BPHGNN
For KDD‘23-submission
> Multiplex Heterogeneous Graph Neural Network with Behavior Pattern Modeling

## Dependencies
Recent versions of the following packages for Python 3 are required:
* numpy==1.21.2
* torch==1.9.1
* scipy==1.7.1
* scikit-learn==0.24.2
* pandas==0.25.0

## Datasets
All of the datasets we use are publicly available datasets.
### Link
* IMDB https://github.com/RuixZh/SR-RSC
* Alibaba https://github.com/xuehansheng/DualHGCN
* Alibaba-s https://github.com/xuehansheng/DualHGCN
* DBLP https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=0
* Taobao https://tianchi.aliyun.com/competition/entrance/231719/information/
* Douban https://github.com/7thsword/MFPR-Datasets

### Preprocess
We compress the data set into a mat format file, which includes the following contents.
* edge: array of subnetworks after coupling, each element in the array is a subnetwork.
* feature: attributes of each node in the network.
* labels: label of labeled points.
* train: index of training set points for node classification. 
* valid: index of validation set points for node classification.
* test: index of test set points for node classification.
* (dataset)_encode: breadth behavior pattern matrix.

In addition, we also sample the positive and negative edges in the network, and divide them into three text files: train, valid and test for link prediction.

## Usage
First, you need to determine the dataset. If you want to do node classification tasks, you need to modify the dataset path in `Node_classification.py`. If you want to do link prediction, you need to modify the dataset path in `Link_prediction.py`.

Second, you need to modify the number of weights in `Model.py`. The number of weights should be the number of behavior pattern after decoupling.

Thirdly, you need to modify  the number of behavior pattern in `Decoupling_matrix_aggregation.py`.

Finally, you need to modify 'node_classfication_evaluate.py' and 'Utils.py' to load the training set, valid set, and test set of the target dataset.

Execute the following command to run the node classification task:

* `python Node_Classfication.py`

Execute the following command to run the link prediction task:

* `python Link_Prediction.py`
