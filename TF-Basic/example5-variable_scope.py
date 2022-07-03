import tensorflow as tf

with tf.name_scope("a_name_scope") as name_scocope:
    var1 = tf.get_variable(name="var1", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
    var2 = tf.Variable(name="var2", initial_value=[2,5], dtype=tf.float32)
    var3 = tf.Variable(name="var3", initial_value=[2.1], dtype=tf.float32)
    var4 = tf.Variable(name="var4", initial_value=[2.2], dtype=tf.float32)


sess = tf.Session()

sess.run(tf.global_variables_initializer())

print(var1.name)
print(sess.run(var1))
print("-----------------\n")
print(var2.name)
print(sess.run(var2))
print("-----------------\n")
print(var3.name)
print(sess.run(var3))
print("-----------------\n")
print(var4.name)
print(sess.run(var4))

with tf.variable_scope("a_variable_scope") as variable_scocope:
    initializer = tf.constant_initializer(value=3)

    var3 = tf.get_variable('var3', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(3.0))
    var4 = tf.get_variable('var4', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(3.0))



sess.run(tf.global_variables_initializer())

#print(var3.name)
#print(sess.run(var3))
#
#
#print(var4.name)
#print(sess.run(var4))



