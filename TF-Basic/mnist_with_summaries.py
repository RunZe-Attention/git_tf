from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None  # 全局变量


# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# 一个通用的用于构建一个layer层节点，且包含张量汇总
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


def train():
    # 加载数据
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                      one_hot=True,
                                      fake_data=FLAGS.fake_data)

    # 这个函数使用了上面的mnist,没有移动到外面
    def feed_dict(train):  # 这个train=true  or  false 不同情况  传入的数据不同
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:  # 训练数据
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)  # 分批次读入
            k = FLAGS.dropout  # 训练的时候drop
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0  # 测试的时候drop固定为1
        return {x: xs, y_: ys, keep_prob: k}

    # 打开会话
    sess = tf.InteractiveSession()

    # 建立网络模型
    # 输入节点
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    # 输入变形，只用于可视化图像
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    # 调用函数生成节点tf.name_scope('layer1') 且汇总里面的张量
    hidden1 = nn_layer(x, 784, 500, 'layer1')

    # dropout节点 并汇总scalar：keep_prob
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

    # Do not apply softmax activation yet, see below.
    # 调用函数生成节点tf.name_scope('layer2') 且汇总里面的张量
    y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

    # 交叉熵节点 里面还有total节点，汇总交scalar:叉熵的均值
    with tf.name_scope('cross_entropy'):
        # The raw formulation of cross-entropy,
        #
        # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
        #                               reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.nn.softmax_cross_entropy_with_logits on the
        # raw outputs of the nn_layer above, and then average across
        # the batch.
        diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    # 训练节点
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            cross_entropy)  # 自适应优化器

    # accuracy节点  里面有两个节点，最后汇总scalar：平均准确率
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # 汇总所有节点
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    # 训练时
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',
                                          sess.graph)  # 这里不仅汇总节点，而且会生成计算图（因为有graph）
    # 测试时
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')  # 仅汇总节点

    # 训练前初始化变量
    tf.global_variables_initializer().run()

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries

    for i in range(FLAGS.max_steps):
        if i % 10 == 0:  # 每10批数据 Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))  # merged是汇总
            test_writer.add_summary(summary, i)  # test_writer实例，将汇总写入test部分
            print('Accuracy at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)  # merged是汇总
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)  # train_writer实例，将汇总写入train部分
                train_writer.add_summary(summary, i)  # train_writer实例，将汇总写入train部分,一定要加上i(即step)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))  # merged是汇总
                train_writer.add_summary(summary, i)  # train_writer实例，将汇总写入train部分,一定要加上i(即step)
    train_writer.close()  # 关闭实例
    test_writer.close()  # 关闭实例


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str, default='../DataSet/mnist/data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='../Tensorboard/mnist',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    print("文件名称是:{}".format(sys.argv[0]))
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

