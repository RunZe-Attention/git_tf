import tensorflow as tf

with tf.variable_scope('scope1'):
    w1 = tf.Variable(1, name='w1')
    w2 = tf.get_variable(name='w2', initializer=2.)
    w3 = tf.Variable(2,name='w1')

with tf.variable_scope('scope1',reuse=True):
    w1_p = tf.Variable(1, name='w1')
    w2_p = tf.get_variable(name='w2', initializer=3.)
    #w3_p = tf.get_variable(name='w2', initializer=3.)

print('w1', w1)
print('w1_p', w1_p)
print("---------------------------\n")
print('w2', w2)
print('w2_p', w2_p)

print("---------------------------\n")
print('w3', w3)
#print('w3_p', w3_p)

print(w1 is w1_p, w2 is w2_p)