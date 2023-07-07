
'''
从大词典中获取特定于于语料的词典
将数据处理成待打标签的形式
'''
import numpy as np
import pickle
from gensim.models import KeyedVectors
# 将 word2vec 文件保存成 binary 格式的文件
def trans_bin(path1, path2):
    wv_from_text = KeyedVectors.load_word2vec_format(path1, binary=False)
    # 如果每次都用上面的方法加载，速度非常慢，可以将词向量文件保存成bin文件，以后就加载bin文件，速度会变快
    wv_from_text.init_sims(replace=True)
    wv_from_text.save(path2)
    '''
    读取用一下代码
    model = KeyedVectors.load(embed_path, mmap='r')
    '''

# 构建新的词典和词向量矩阵
def get_new_dict(type_vec_path, type_word_path, final_vec_path, final_word_path):
    # 原词 159018 找到的词 133959 找不到的词 25059
    # 添加 unk 过后 159019 找到的词 133960 找不到的词 25059
    # 添加 pad 过后 词典：133961 词向量 133961
    # 加载转换文件
    model = KeyedVectors.load(type_vec_path, mmap='r')

    with open(type_word_path, 'r') as f:
        total_word = eval(f.read())
        f.close()

    # 输出词向量
    word_dict = ['PAD', 'SOS', 'EOS', 'UNK']  # 其中 0 PAD_ID, 1 SOS_ID, 2 E0S_ID, 3 UNK_ID

    fail_word = []
    rng = np.random.RandomState(None)
    pad_embedding = np.zeros(shape=(1, 300)).squeeze()
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    sos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    eos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    word_vectors = [pad_embedding, sos_embedding, eos_embedding, unk_embedding]
    print(len(total_word))
    for word in total_word:
        try:
            word_vectors.append(model.wv[word])  # 加载词向量
            word_dict.append(word)
        except:
            print(word)
            fail_word.append(word)
    
    # 关于有多少个词，以及多少个词没有找到
    print(len(word_dict))
    print(len(word_vectors))
    print(len(fail_word))

    word_vectors = np.array(word_vectors)
    word_dict = dict(map(reversed, enumerate(word_dict)))
    
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)

    with open(final_vec_path, 'rb') as file:
        v = pickle.load(file)

    with open(final_word_path, 'rb') as f:
        word_dict = pickle.load(f)

    print("完成")

# 得到词在词典中的位置
def get_index(type, text, word_dict):
    location = []

    if type == 'code':
        location.append(1)
        len_c = len(text)

        if len_c + 1 < 350:
            if len_c == 1 and text[0] == '-1000':
                location.append(2)
            else:
                for i in range(0, len_c):
                    if word_dict.get(text[i]) is not None:
                        index = word_dict.get(text[i])
                    else:
                        index = word_dict.get('UNK')
                    location.append(index)

                location.append(2)
        else:
            for i in range(0, 348):
                if word_dict.get(text[i]) is not None:
                    index = word_dict.get(text[i])
                else:
                    index = word_dict.get('UNK')
                location.append(index)

            location.append(2)
    else:
        if len(text) == 0:
            location.append(0)
        elif text[0] == '-10000':
            location.append(0)
        else:
            for i in range(0, len(text)):
                if word_dict.get(text[i]) is not None:
                    index = word_dict.get(text[i])
                else:
                    index = word_dict.get('UNK')
                location.append(index)

    return location

#将训练、测试、验证语料序列化
def Serialization(word_dict_path, type_path, final_type_path):
    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)

    with open(type_path, 'r') as f:
        corpus = eval(f.read())

    total_data = []

    for i in range(len(corpus)):
        qid = corpus[i][0]
        Si_word_list = get_index('text', corpus[i][1][0], word_dict)
        Si1_word_list = get_index('text', corpus[i][1][1], word_dict)
        tokenized_code = get_index('code', corpus[i][2][0], word_dict) 
        query_word_list = get_index('text', corpus[i][3], word_dict)

        block_length = 4
        label = 0

        if len(Si_word_list) > 100:
            Si_word_list = Si_word_list[:100]
        else:
            Si_word_list += [0] * (100 - len(Si_word_list))

        if len(Si1_word_list) > 100:
            Si1_word_list = Si1_word_list[:100]
        else:
            Si1_word_list += [0] * (100 - len(Si1_word_list))

        if len(tokenized_code) < 350:
            tokenized_code += [0] * (350 - len(tokenized_code))
        else:
            tokenized_code = tokenized_code[:350]

        if len(query_word_list) > 25:
            query_word_list = query_word_list[:25]
        else:
            query_word_list += [0] * (25 - len(query_word_list))

        one_data = [qid, [Si_word_list, Si1_word_list], [tokenized_code], 
                    query_word_list, block_length, label]
        total_data.append(one_data)

    with open(final_type_path, 'wb') as file:
        pickle.dump(total_data, file)


def get_new_dict_append(type_vec_path, previous_dict, previous_vec, append_word_path, final_vec_path, final_word_path):
    # 加载转换文件
    model = KeyedVectors.load(type_vec_path, mmap='r')

    with open(previous_dict, 'rb') as f:
        pre_word_dict = pickle.load(f)

    with open(previous_vec, 'rb') as f:
        pre_word_vec = pickle.load(f)

    with open(append_word_path, 'r') as f:
        append_word = eval(f.read())
        f.close()

    # 输出词向量
    print(type(pre_word_vec))
    word_dict = list(pre_word_dict.keys())  # '其中0 PAD_ID,1SOS_ID,2E0S_ID,3UNK_ID
    print(len(word_dict))
    word_vectors = pre_word_vec.tolist()
    print(word_dict[:100])
    fail_word = []
    print(len(append_word))
    rng = np.random.RandomState(None)
    unk_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
 
    # 遍历要追加的词，找到词向量并加入原词向量列表和词典中
    for word in append_word:
        try:
            word_vectors.append(model.wv[word])
            word_dict.append(word)
        except:
            fail_word.append(word)
    # 关于有多少个词，以及多少个词没有找到
    print(len(word_dict))
    print(len(word_vectors))
    print(len(fail_word))
    print(word_dict[:100])
    word_vectors = np.array(word_vectors)
    # print(word_vectors.shape)
    word_dict = dict(map(reversed, enumerate(word_dict)))
    
    # 写入文件
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)

    print("完成")



def preprocess_ps_data(ps_path, ps_path_bin):
    trans_bin(ps_path, ps_path_bin)

def preprocess_sql_data(sql_path, sql_path_bin):
    trans_bin(sql_path, sql_path_bin)

def create_python_word_dict(ps_path_bin, python_word_path, python_word_vec_path, python_word_dict_path):
    get_new_dict(ps_path_bin,python_word_path,python_word_vec_path,python_word_dict_path)

def create_sql_word_dict(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path):
    get_new_dict(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path)
    
def preprocess_and_create_word_dict(ps_path, ps_path_bin, sql_path, sql_path_bin, python_word_path, python_word_vec_path, python_word_dict_path, sql_word_path, sql_word_vec_path, sql_word_dict_path):
    preprocess_ps_data(ps_path, ps_path_bin)
    preprocess_sql_data(sql_path, sql_path_bin)
    create_python_word_dict(ps_path_bin, python_word_path, python_word_vec_path, python_word_dict_path)
    create_sql_word_dict(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path)

def create_label_data(new_sql_staqc, sql_final_word_dict_path, new_sql_large, large_sql_f, new_python_staqc, python_final_word_dict_path, new_python_large,large_python_f,staqc_sql_f,staqc_python_f):
    Serialization(sql_final_word_dict_path, new_sql_staqc, staqc_sql_f)
    Serialization(sql_final_word_dict_path, new_sql_large, large_sql_f)
    Serialization(python_final_word_dict_path, new_python_staqc, staqc_python_f)
    Serialization(python_final_word_dict_path, new_python_large, large_python_f)

if __name__ == '__main__':
    # define file paths
    ps_path = '../hnn_process/embeddings/10_10/python_struc2vec1/data/python_struc2vec.txt' #239s
    ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin' #2s

    sql_path = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.txt'
    sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'

    python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'
    python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'
    python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'

    sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'
    sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'
    sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'

    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'

    sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'
    sql_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'

    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    final_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    python_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
    python_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl'
    
    # 数据预处理并创建词典
    preprocess_and_create_word_dict(ps_path, ps_path_bin, sql_path, sql_path_bin, python_word_path, python_word_vec_path, python_word_dict_path, sql_word_path, sql_word_vec_path, sql_word_dict_path)

    # 创建标签数据
    create_label_data(new_sql_staqc, sql_final_word_dict_path, new_sql_large, "large_sql_f", new_python_staqc, python_final_word_dict_path, new_python_large, "large_python_f", "staqc_sql_f", "staqc_python_f")


    print('序列化完毕')






