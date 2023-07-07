import pickle


def get_vocab(corpus1, corpus2):
    """
    :param corpus1: (list)文本数据列表1
    :param corpus2: (list)文本数据列表2
    :return: 词表集合(set)
    """
    word_vocab = set()
    for corpus in [corpus1, corpus2]:
        for record in corpus:
            for i in range(1, 4):
                for j in range(len(record[i])):
                    for k in range(len(record[i][j])):
                        word_vocab.add(record[i][j][k])
    print(len(word_vocab))
    return word_vocab


def load_pickle(filename):
    """
    :param filename:(str)文件名
    :return: 反序列化后的对象
    """
    return pickle.load(open(filename, 'rb'), encoding='iso-8859-1')


def write_file(filename, content):
    """
    :param filename:(str)文件路径和文件名
    :param content:(str)要写的内容
    :return:
    """
    with open(filename, "w") as f:
        f.write(str(content))


def vocab_prpcessing(filepath1, filepath2, save_path):
    """
    :param filepath1:(str)原始数据文件1的路径
    :param filepath2:(str)原始数据文件2的路径
    :param save_path:(str)保存词表的路径
    :return:
    """
    with open(filepath1, 'r')as f:
        total_data1 = eval(f.read())
        f.close()

    with open(filepath2, 'r')as f:
        total_data2 = eval(f.read())
        f.close()

    word_set = get_vocab(total_data1, total_data2)
    write_file(save_path, word_set)


def final_vocab_prpcessing(filepath1, filepath2, save_path):
    """
    :param filepath1:(str)已有词表文件的路径
    :param filepath2:(str)原始数据文件的路径
    :param save_path:(str)保存新词表的路径
    :return:
    """
    with open(filepath1, 'r') as f:
        total_data1 = set(eval(f.read()))
        f.close()

    with open(filepath2, 'r') as f:
        total_data2 = eval(f.read())
        f.close()

    x1 = get_vocab(total_data1, total_data2)
    word_set = x1 - total_data1
    
    print(len(total_data1))
    print(len(word_set))
    
    write_file(save_path, word_set)


if __name__ == "__main__":
    #====================获取staqc的词语集合===============
    python_hnn = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/python_hnn_data_teacher.txt'
    python_staqc = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/python_staqc_data.txt'
    python_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/python_word_vocab_dict.txt'

    sql_hnn = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/sql_hnn_data_teacher.txt'
    sql_staqc = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/sql_staqc_data.txt'
    sql_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/sql_word_vocab_dict.txt'


    #====================获取最后大语料的词语集合的词语集合===============
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'
    
    final_vocab_prpcessing(sql_word_dict, new_sql_large, large_word_dict_sql)
    

    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large ='../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    
    
