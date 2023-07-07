from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import random
import pickle
import argparse
import logging
from sklearn.metrics import *
from configs import *
import warnings



random.seed(42)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
warnings.filterwarnings("ignore")

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)




class StandaloneCode:
    def init(self, conf=None):
        self.conf = conf or {}
        self._buckets = self.conf.get('buckets', [(2, 10, 22, 72), (2, 20, 34, 102), (2, 40, 34, 202), (2, 100, 34, 302)])
        self._buckets_text_max = (max(i for i, _, _, _ in self._buckets), max(j for _, j, _, _ in self._buckets))
        self._buckets_code_max = (max(i for _, _, i, _ in self._buckets), max(j for _, _, _, j in self._buckets))
        self.path = self.conf.get('workdir', './data/')
        self.train_params = self.conf.get('training_params', {})
        self.data_params = self.conf.get('data_params', {})
        self.model_params = self.conf.get('model_params', {})

def load_pickle(self, filename):
    with open(filename, 'rb') as f:
        word_dict = pickle.load(f)
    return word_dict

def pad(self, data, maxlen):
    return pad_sequences(data, maxlen=maxlen, padding='post', truncating='post', value=0)

def save_model_epoch(self, model, epoch, d12, d3, d4, d5, r):
    model.save("{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(
        self.path, self.model_params['model_name'], d12, d3, d4, d5, r, epoch), overwrite=True)

def load_model_epoch(self, model, epoch, d12, d3, d4, d5, r):
    filepath = "{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(
        self.path, d12, d3, d4, d5, r, epoch)
    assert os.path.exists(filepath), f"Weights at epoch {epoch} not found"
    model.load(filepath)

def del_pre_model(self, prepoch, d12, d3, d4, d5, r):
    if len(prepoch) >= 2:
        lenth = len(prepoch)
        epoch = prepoch[lenth-2]
        filepath = f"{self.path}models/{self.model_params['model_name']}/pysparams:d12={d12}_d3={d3}_d4={d4}_d5={d5}_r={r}_epo={epoch}_class.h5"
        if os.path.exists(filepath):
            os.remove(filepath)

def process_instance(self, instance, target, maxlen):
    target.append(self.pad(instance, maxlen))

def process_matrix(self, inputs, trans1_length, maxlen):
    inputs_trans1 = np.split(inputs, trans1_length, axis=1)
    processed_inputs = [np.squeeze(item, axis=1).tolist() for item in inputs_trans1]
    return processed_inputs

def get_data(self, path):
    data = self.load_pickle(path)
    text_S1, text_S2, code, queries, labels, ids = [], [], [], [], [], []

    text_block_length, text_word_length, query_word_length, code_token_length = 2, 100, 25, 350
    text_blocks = self.process_matrix(np.array([samples_term[1] for samples_term in data]),
                                      text_block_length, 100)

    text_S1 = text_blocks[0]
    text_S2 = text_blocks[1]

    code_blocks = self.process_matrix(np.array([samples_term[2] for samples_term in data]),
                                      text_block_length - 1, 350)
    code = code_blocks[0]

    queries = [samples_term[3] for samples_term in data]
    labels = [samples_term[5] for samples_term in data]
    ids = [samples_term[0] for samples_term in data]

    return text_S1, text_S2, code, queries, labels, ids

def eval(self, model, path):
    text_S1, text_S2, code, queries, labels, ids = self.get_data(path)

    labelpred = model.predict([np.array(text_S1), np.array(text_S2), np.array(code), np.array(queries)],
                              batch_size=100)
    labelpred = np.argmax(labelpred, axis=1)

    loss = log_loss(labels, labelpred)
    acc = accuracy_score(labels, labelpred)
    f1 = f1_score(labels, labelpred)
    recall = recall_score(labels, labelpred)
    precision = precision_score(labels, labelpred)

    print(f"相应的测试性能: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}, accuracy={accuracy:.3f}")

    return acc, f1, recall, precision, loss

def u2l_codemf(self, model, path, save_path):
    total_label = []
    text_S1, text_S2, code, queries, labels, ids1 = self.get_data(path)

    labelpred = model.predict([np.array(text_S1), np.array(text_S2), np.array(code), np.array(queries)],
                              batch_size=100)
    labelpred1 = np.argmax(labelpred, axis=1)

    total_label.append(ids1)
    total_label.append(labelpred1.tolist())
    with open(save_path, "w") as f:
        f.write(str(total_label))
    print("codemf标签已打完")

def u2l_textsa(self, model, path, save_path):
    with open(save_path, 'r') as f:
        pre = eval(f.read())

    my_pre1 = pre[1]  # codemf_label
    total_label = []
    text_S1, text_S2, code, queries, labels, ids1 = self.get_data(path)
    labelpred = model.predict([np.array(text_S1), np.array(text_S2), np.array(code), np.array(queries)],
                              batch_size=100)
    labelpred1 = np.argmax(labelpred, axis=1)

    total_label.append(ids1)
    total_label.append(my_pre1)
    total_label.append(labelpred1.tolist())
    with open(save_path, "w") as f:
        f.write(str(total_label))
    print("textsa标签已打完")

def u2l_codesa(self, model, path, save_path):
    with open(save_path, 'r') as f:
        pre = eval(f.read())

    my_pre1 = pre[1]  # codemf_label
    my_pre2 = pre[2]  # textsa_label

    total_label = []
    text_S1, text_S2, code, queries, labels, ids1 = self.get_data(path)
    labelpred = model.predict([np.array(text_S1), np.array(text_S2), np.array(code), np.array(queries)],
                              batch_size=100)
    labelpred1 = np.argmax(labelpred, axis=1)

    total_label.append(ids1)
    total_label.append(my_pre1)
    total_label.append(my_pre2)
    total_label.append(labelpred1.tolist())
    with open(save_path, "w") as f:
        f.write(str(total_label))
    print("codesa标签已打完")


#分析组合不同模型打标签的结果
'''
这一步是已经确定了选择text_sa与code_sa中的模型,与codemf模型标签节后进行最后的标签过滤
'''


def final_analysis(path, hnn_path, save_path):
    with open(path, 'r') as f:
        pre = eval(f.read())
        ids = pre[0]
        codemf_label = pre[1]
        textsa_label = pre[2]
        codesa_label = pre[3]
        f.close()

    hnn_lable_1 = []
    with open(hnn_path, 'r') as f:
        hnn = eval(f.read())
        for i, item in enumerate(hnn[0]):
            if hnn[1][i] == 1:
                hnn_lable_1.append(item)
        f.close()

    total_final = []
    for i, item in enumerate(ids):
        if codemf_label[i] == 1 and textsa_label[i] == 1 and codesa_label[i] == 1:
            if item in hnn[0]:
                continue
            else:
                total_final.append(item)

    total_final += hnn_lable_1
    with open(save_path, "w") as f:
        f.write('\n'.join(map(str, total_final)))


def parse_args():
    parser = argparse.ArgumentParser("Train and Test Model")
    parser.add_argument("--train", choices=["python", "sql"], default="sql", help="train dataset set")
    parser.add_argument("--mode", choices=["train", "eval"], default='eval',
                        help="The mode to run. The `train` mode trains a model;"
                             " the `eval` mode evaluat models in a test set ")
    parser.add_argument("--verbose", action="store_true", default=True, help="Be verbose")
    return parser.parse_args()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', default=False, action='store_true', help='Training mode')
    parser.add_argument('--eval', dest='mode', default='train', choices=['train', 'eval'], help=' Eval mode')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    conf = get_config_u2l(train=args.train)
    data_params = conf.get('data_params')
    model_params = conf.get('model_params')
    train_path = data_params.get('train_path')
    dev_path = data_params.get('valid_path')
    test_path = data_params.get('test_path')
    embding = data_params.get('code_pretrain_emb_path')

    logger.info('Build Model')
    model = CARLCS_CNN(conf)
    StandoneCode = StandoneCode(conf)

    # sql 标签路径
    staqc_sql_f = '../data_processing/hnn_process/ulabel_data/staqc/seri_sql_staqc_unlabled_data.pkl'
    large_sql_f = '../data_processing/hnn_process/ulabel_data/large_corpus/multiple/seri_ql_large_multiple_unlable.pkl'
    large_single_path = '../data_processing/hnn_process/ulabel_data/large_corpus/single/sql_large_single_label.txt'
    hnn_lable_sql_path = '../data_processing/hnn_process/ulabel_data/staqc/hnn_label_sql.txt'
    staqc_sql_final_label = '../data_processing/hnn_process/ulabel_data/staqc/sql_final_label.txt'
    save_path_final_lable_staqc_sql = '../data_processing/hnn_process/ulabel_data/staqc/combine_final_sql_lable.txt'

    # large corpus python 标签路径
    staqc_python_f = '../data_processing/hnn_process/ulabel_data/staqc/seri_python_staqc_unlabled_data.pkl'
    large_python_f = '../data_processing/hnn_process/ulabel_data/large_corpus/multiple/seri_python_large_multiple_unlable.pkl'
    large_single_python_path ='../data_processing/hnn_process/ulabel_data/large_corpus/single/python_large_single_label.txt'
    hnn_lable_python_path = '../data_processing/hnn_process/ulabel_data/staqc/hnn_label_python.txt'
    staqc_python_final_lable ='../data_processing/hnn_process/ulabel_data/staqc/python_final_label.txt'
    save_path_final_lable_staqc_python = '../data_processing/hnn_process/ulabel_data/staqc/combine_final_python_lable.txt'

    # 调整参数
    drop1 = drop2 = drop3 = drop4 = drop5 = np.round(0.25, 2)
    model.params_adjust(dropout1=drop1, dropout2=drop2, dropout3=drop3, dropout4=drop4, dropout5=drop5,
                        Regularizer=round(0.0004, 4), num=8, seed=42)

    model.build()
    if args.mode == 'eval':
        StandoneCode.load_model_epoch(model, 1111, 0.1, 0.1, 0.1, 0.1, 101)
        final_analay(staqc_sql_final_label, hnn_lable_sql_path, save_path_final_lable_staqc_sql)

