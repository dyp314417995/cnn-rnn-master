# coding: utf-8

import sys
from collections import Counter

import numpy as np
import importlib
import tensorflow.contrib.keras as kr

print(sys.version_info)
# 下边的代码用来判断python的版本
# reload(sys)是py2的写法，py3中用importlib.reload(sys)代替
if sys.version_info[0] > 2:
    is_py3 = True
else:
    # reload(sys)
    importlib.reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


# 读取文件数据;


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                # 去除空白后按Tab分隔
                label, content = line.strip().split('\t')
                # 在python中，非0 和非null都是为真(True)的，只有0和null才为假(False)
                if content:
                    # 这里为什么第一个用了list函数，第二个没有？
                    # label本身就是集合吗？这里有疑问
                    # list() 方法用于将元组转换为列表
                    # 注：元组与列表是非常类似的，区别在于元组的元素值不能修改，元组是放在括号中，列表是放于方括号中。
                    contents.append(list(native_content(content)))
                    # labels.append(list(native_content(label)))
                    labels.append(native_content(label))

            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


# category 类别
def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)
    # 创建数据列表和标签列表
    data_id, label_id = [], []
    # range() 函数可创建一个整数列表
    # 循环文本长度次
    for i in range(len(contents)):
        # 列表推导式就是利用列表创建新列表:就是利用for循环迭代一个列表，然后用if条件筛选出符合条件的数据变成一个新的列表
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    print(filename + "data_id" + data_id)
    print(filename + "label_id" + label_id)
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    # num_classes ：total number of classes
    num_classes = len(cat_to_id)
    # to_categorical: 将类向量（整数）转换为二进制类矩阵
    y_pad = kr.utils.to_categorical(label_id, num_classes)  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x_train, y_train, batch_size):
    """生成批次数据"""

    data_len = len(x_train)
    print("data_len" + data_len)
    num_batch = int((data_len - 1) / batch_size) + 1
    # np:numpy
    # 传给permutation一个矩阵，它会返回一个洗牌后的矩阵副本
    indices = np.random.permutation(np.arange(data_len))
    print("indices:"+indices)
    x_shuffle = x_train[indices]
    y_shuffle = y_train[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
