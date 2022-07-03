
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
print('check：libs well prepared')

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename,expected_bytes):
    #判断文件是否存在
    if not os.path.exists(filename):
        #下载
        print('download...')
        filename, _ = urlretrieve(url + filename,filename)
    #校验大小
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print('exception %s' % statinfo.st_size)
    return filename

filename = maybe_download('text8.zip',31244016)
print(filename)


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        #tf.compat.as_str 数据转单词列表
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)
print('Data size %d' % len(words))

vocabulary_size = 50000


def build_dataset(words):
    # 将所有低频单词设为UNK，个数先设为-1
    count = [['UNK', -1]]
    # 将words集合中的单词按频数排序，将频率最高的前vocabulary_size-1个单词以及他们的出现的个数按顺序输出到count中，
    # 将频数排在n_words-1之后的单词设为UNK。同时，count的规律为索引越小，单词出现的频率越高
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    # 构建字典单词到数字的映射
    for word, _ in count:
        # 对count中所有单词进行编号，赋予ID，由0开始，保存在字典dict中
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0

    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)

    # 记录UNK个数
    count[0][1] = unk_count

    # 数字到单词的映射
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


# 映射之后的训练数据
data, count, dictionary, reverse_dictionary = build_dataset(words)

print('Most common words (+UNK)', count[:5])
print('original data', words[:10])
print('training data', data[:10])


# 这个函数的功能是对数据data中的每个单词，分别与前1个单词和后1个个单词生成一个batch
# skip_window代表左右各选取词的个数，num_skips代表预测周围单词的总数
def generate_batch(batch_size, num_skips, skip_window):
    # 定义全局变量
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    # x y、
    # 建一个大小为batch的数组
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    # 建一个（batch，1）大小的二位数组，保存任意单词前一个或者后一个单词，从而形成一个pair
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # 窗的大小，为3，结构为[ skip_window target skip_window ]
    span = 2 * skip_window + 1
    # 建立一个结构为双向队列的缓冲区，大小不超过span
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        # 循环使用
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        # 将target赋值为1，即当前单词
        target = skip_window
        # 将target存入targets_to_avoid中，避免重复存入
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            # 选出还没出现在targets_to_avoid中的单词索引
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            # 存入targets_to_avoid
            targets_to_avoid.append(target)
            # 在batch中存入当前单词
            batch[i * num_skips + j] = buffer[skip_window]
            # 在labels中存入当前单词前一个单词或者后一个单词
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


print('data:', [reverse_dictionary[di] for di in data[:8]])
data_index = 0
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=2)
print('batch:', [reverse_dictionary[bi] for bi in batch])
print('labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_example = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    # 输入数据
    # 输入一个batch的训练数据，是当前单词在字典中的索引id
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    # 输入一个batch的训练数据的标签，是当前单词前一个或者后一个单词在字典中的索引id
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    # 从字典前100个单词，即频率最高的前100个单词中，随机选出16个单词，将它们的id储存起来，作为验证集
    valid_dataset = tf.constant(valid_example, tf.int32)

    # 初始化变量字典中每个单词的embeddings，值为-1到1的均匀分布
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # 初始化训练参数
    softmax_weight = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    softmax_bisase = tf.Variable(tf.zeros([vocabulary_size]))

    # 本次训练数据对应的embedding
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    # batch loss
    # Compute the average loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels eachtime we evaluate the loss.

    # 根据词频或者类似词频的概率选出64个负采样v，联同正确的输入w（都是词的id），用它们在weights对应的向量组成
    # 一个训练子集。对于训练子集中各个元素，如果是w或者m(i)==w(w这里是输入对应的embedding)，
    # loss(i)=log(sigmoid(w*mu(i)))如果是负采样，则loss(i)=log(1-sigmoid(w*mu(i)))然后将所有loss加起来作为总的
    # loss，loss越小越相似（余弦定理）用总的loss对各个参数求导数，来更新weight以及输入的embedding

    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weight,
                                                     biases=softmax_bisase,
                                                     inputs=embed,
                                                     labels=train_labels,
                                                     num_sampled=num_sampled,
                                                     num_classes=vocabulary_size))
    # 优化loss,更新参数
    optimizer = tf.train.AdamOptimizer(1.0).minimize(loss)

    # 归一化
    # 调用reduce_sum(arg1, arg2)时，参数arg1即为要求和的数据，arg2有两个取值分别为0和1，通常用reduction_indices=[0]或
    # reduction_indices=[1]来传递参数。从上图可以看出，当arg2 = 0时，是纵向对矩阵求和，原来矩阵有几列就得到几个值；相
    # 似地，当arg2 = 1时，是横向对矩阵求和；当省略arg2参数时，默认对矩阵所有元素进行求和。
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))

    normalized_embeddings = embeddings / norm

    # 用已有的embedding计算valid的相似词
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

num_steps = 100000
with tf.Session(graph=graph) as session:
    # Add variable initializer.
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps + 1):
        # 生成一个batch的训练数据
        batch_data, batch_labels = generate_batch(batch_size, num_skips, skip_window)

        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        # 每2000出打印一次平均loss
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0

        # 打印valid效果
        if step % 10000 == 0:
            # 每10000步评估一下验证集和整个embeddings的相似性
            # 结果是验证集中每个词和字典中所有词的相似性
            sim = similarity.eval()
            for i in range(valid_size): 
                # 根据id找回词
                valid_word = reverse_dictionary[valid_example[i]]
                # 相似度最高的5个词
                top_k = 5
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)

        final_embeddings = normalized_embeddings.eval()







