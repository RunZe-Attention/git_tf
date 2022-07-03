
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '../DataSet/mnist/data', 'dataset path')
#tf.app.flags.DEFINE_float('learning_rate', 0.0001, '''初始学习率''')
#tf.app.flags.DEFINE_integer('train_steps', 50000, '''总的训练轮数''')
#tf.app.flags.DEFINE_boolean('is_use_gpu', False, '''是否使用GPU''')


print('模型保存路径： {}'.format(FLAGS.data_dir))
# 加载数据
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)


# 定义回归模型
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10]) # 输入的真实值的占位符

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, w) + b # 预测值

a = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
        print("训练:{}".format(i+1))


        if (i+1 )% 100 == 0:
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("使用测试集训练的精度为:{}".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))


