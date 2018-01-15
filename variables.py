#对variable创建变量进行测试

# 载入模块
import numpy as np
import tensorflow as tf

sess = tf.Session()

############test 1##################

#创建w变量
w = tf.Variable(3)
#创建assign_op操作
assign_op1 = w.assign(10)

#对w进行初始化
sess.run(w.initializer)
#输出w
print(sess.run(w)) # > 3
#进行assign_op操作后输出
print(sess.run(assign_op1)) # >10

###############test 2###############
#定义操作assign_op2
assign_op2 = w.assign(2*w)

#重新进行全局初始化，此时w=3
sess.run(tf.global_variables_initializer())

print(sess.run(assign_op2)) # >> 6
print(sess.run(assign_op2)) # >> 12
print(sess.run(assign_op2)) # >> 24

sess.close()

#############test 3#############
#不同的session不会相互干扰
#定义两个Session
sess1=tf.Session()
sess2=tf.Session()

#两个session分别对w进行初始化
sess1.run(w.initializer)
sess2.run(w.initializer)

print(sess1.run(w.assign_add(10)))  # >> 13
print(sess2.run(w.assign_sub(1)))   # >> 2

print(sess1.run(w.assign_add(10)))  # >> 23
print(sess2.run(w.assign_sub(1)))   # >> 1

sess1.close()
sess2.close()