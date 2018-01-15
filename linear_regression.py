import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xlrd
import utils

# data file path
data_file = 'data/fire_theft.xls'

# step1:read in data from data file
# open an excel file to read data
book = xlrd.open_workbook(data_file, encoding_override="utf-8")
# Accroding sheet index to get
sheet = book.sheet_by_index(0)
# change sheet to array
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
# get rows
num = sheet.nrows - 1

# step2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# step3: create weight and bias, and initializer to 0.0
w1 = tf.Variable(0.0, name='weight1')
w2 = tf.Variable(0.0, name='weight2')
w3 = tf.Variable(0.0, name='weight3')
b = tf.Variable(0.0, name='bias')

# step4: build model to predict Y
Y_predicted = (X**3) * w1 + (X**2) * w2 + X * w3 + b

# step5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')

# step6: using gradient descent with learning rate of 0.0001 to minimize loss
optimizer = tf.train.AdamOptimizer(
    learning_rate=0.0001).minimize(loss)

with tf.Session() as sess:
    # step7: initializer the necessary variables w and b
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('D://logs', sess.graph)

    # step8: train the model
    for i in range(100):
        total_loss = 0

        for x, y in data:
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += l

        print('Epoch {0}: {1}'.format(i, total_loss / num))

    writer.close()

    # step9: output the values of w and b
    w1, w2, w3, b = sess.run([w1, w2, w3, b])
print(w1, w2, w3, b)
# step10 : plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')

X = np.linspace(0, 40, 1000)
plt.plot(X, (X**3) * w1 + (X**2) * w2 + X *
         w3 + b, 'r', label='Predicted results')
plt.legend()
plt.show()