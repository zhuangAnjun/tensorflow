# 载入tensorflow模块
import tensorflow as tf

#例子1，对占位符赋值

# 创建一个占位符，
a = tf.placeholder(tf.float32, shape=[3])

# 创建一个常量
b = tf.constant([1,1,1], tf.float32)

# 创建一个操作
c = a+b

with tf.Session() as sess:
    # 输出运行结果
    print(sess.run(c, {a: [1,2,3]})) # >>[2. 3. 4.]

#例子2，对变量赋值
d = tf.add(1,2)

e = tf.multiply(d,5)

with tf.Session() as sess:
    #
    print(sess.run(e,{d: 4})) # >>20