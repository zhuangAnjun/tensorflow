"""lazy loading 是指在运行时才定义的操作，此操作每次运行都会创建一个节
点，会使图变复杂导致性能下降，所以一般尽量避免使用lazy loadin
"""
import tensorflow as tf

x = tf.Variable(10, name='x')
y = tf.Variable(15, name='y')
z = tf.add(x,y)

with tf.Session() as sess:
    #进行全局初始化
    sess.run(tf.global_variables_initializer())
    #将将产生的日志输出到"D://logs"目录中
    writer=tf.summary.FileWriter('D://logs', sess.graph)

    #循环运行5次，查看生成几个节点
    for _ in range(5):
        sess.run(z)

    #输出图信息
    print(tf.get_default_graph().as_graph_def())

    writer.close()

with tf.Session() as sess:
    #进行全局初始化变量
    sess.run(tf.global_variables_initializer())

    #将产生的日志输出到"D://logs"目录中
    writer=tf.summary.FileWriter('D://logs', sess.graph)
    #循环5次，查看生成的图有几个节点

    for _ in range(5):
        sess.run(tf.add(x,y))
    #输出图信息
    print(tf.get_default_graph().as_graph_def())

    writer.close();