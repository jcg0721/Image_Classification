import tensorflow as tf
import numpy as np
import threading
import os
from PIL import Image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RESIZE_PATH = "ResizeImage6000\\"

f = open("labeling6000.txt", 'r')
learning_rate = 0.0001
TRAINING_DATA_RATE = 0.9
IMAGE_SIZE = 224
IMAGE_LAYER = 3
CLASS_LABEL = 4
TRAINING_EPOCHS = 10
BATCH_SIZE = 64


def oneHotEncoding(imageLabelList):
    value = np.eye(CLASS_LABEL)[np.array(imageLabelList).reshape(-1)]
    return value


def ImageData(imageNameList):
    image = []
    for name in imageNameList:
        imageFile = Image.open(RESIZE_PATH + name)
        imageRGBArray = np.array(imageFile)
        imageRGBList = imageRGBArray.tolist()
        image.append(imageRGBList)
    return image

def ImageName():
    Score = []
    Name = []

    i = 0
    while True:

        line = f.readline()
        if line == "":
            break
        line = line.split(" ")
        Name.append(line[0])
        Score.append([int(line[1][:-1])])
        i += 1
    return Name, Score

data = [None]
def get_data(data_ref):
    data_ref[0] = input("Info: ")

Name, Score = ImageName()

training_Set = int(len(Name) * TRAINING_DATA_RATE)
test_Set = len(Name) - training_Set

testName = Name[training_Set:]
testScore = Score[training_Set:]
trainingName = Name[:training_Set]
trainingScore = Score[:training_Set]



def conv2d(x, W, b, strid=1, padd='SAME'):
    x = tf.nn.conv2d(x, W, strides=[1, strid, strid, 1], padding=padd)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, s=2, k=2):
    return tf.nn.max_pool(x, ksize=[1, s, s, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(x, weights, biases, dropout):
    conv1_1 = conv2d(x, weights['wc1_1'], biases['bc1_1'])
    conv1_2 = conv2d(conv1_1, weights['wc1_2'], biases['bc1_2'])
    pool1 = maxpool2d(conv1_2)

    conv2_1 = conv2d(pool1, weights['wc2_1'], biases['bc2_1'])
    conv2_2 = conv2d(conv2_1, weights['wc2_2'], biases['bc2_2'])
    pool2 = maxpool2d(conv2_2)

    conv3_1 = conv2d(pool2, weights['wc3_1'], biases['bc3_1'])
    conv3_2 = conv2d(conv3_1, weights['wc3_2'], biases['bc3_2'])
    conv3_3 = conv2d(conv3_2, weights['wc3_3'], biases['bc3_3'])
    pool3 = maxpool2d(conv3_3)

    conv4_1 = conv2d(pool3, weights['wc4_1'], biases['bc4_1'])
    conv4_2 = conv2d(conv4_1, weights['wc4_2'], biases['bc4_2'])
    conv4_3 = conv2d(conv4_2, weights['wc4_3'], biases['bc4_3'])
    pool4 = maxpool2d(conv4_3)

    conv5_1 = conv2d(pool4, weights['wc5_1'], biases['bc5_1'])
    conv5_2 = conv2d(conv5_1, weights['wc5_2'], biases['bc5_2'])
    conv5_3 = conv2d(conv5_2, weights['wc5_3'], biases['bc5_3'], strid=2)
    pool5 = maxpool2d(conv5_3)

    fc1 = tf.reshape(pool5, [-1, 8192])
    fc1 = tf.nn.relu(tf.matmul(fc1, weights['wc6']) + biases['bc6'])
    fc1 = tf.nn.dropout(fc1, keep_prob=dropout)

    fc2 = tf.nn.relu(tf.matmul(fc1, weights['wc7']) + biases['bc7'])
    fc2 = tf.nn.dropout(fc2, keep_prob=dropout)

    fc3 = tf.nn.relu(tf.matmul(fc2, weights['wc8']) + biases['bc8'])
    fc3 = tf.nn.dropout(fc3, keep_prob=dropout)

    fc4 = tf.add(tf.matmul(fc3, weights['wc9']), biases['bc9'])
    return fc4

weights = {
    'wc1_1': tf.Variable(tf.random_normal([3, 3, 3, 64], stddev=0.01)),
    'wc1_2': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01)),
    'wc2_1': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01)),
    'wc2_2': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.01)),
    'wc3_1': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01)),
    'wc3_2': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.01)),
    'wc3_3': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.01)),
    'wc4_1': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01)),
    'wc4_2': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01)),
    'wc4_3': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01)),
    'wc5_1': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01)),
    'wc5_2': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01)),
    'wc5_3': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01)),
    'wc6': tf.get_variable("W6", shape=[8192, 1800], initializer=tf.contrib.layers.xavier_initializer()),
    'wc7': tf.get_variable("W7", shape=[1800, 300], initializer=tf.contrib.layers.xavier_initializer()),
    'wc8': tf.get_variable("W8", shape=[300, 60], initializer=tf.contrib.layers.xavier_initializer()),
    'wc9': tf.get_variable("W9", shape=[60, 4], initializer=tf.contrib.layers.xavier_initializer())
}

biases = {
    'bc1_1': tf.Variable(tf.random_normal([64])),
    'bc1_2': tf.Variable(tf.random_normal([64])),
    'bc2_1': tf.Variable(tf.random_normal([128])),
    'bc2_2': tf.Variable(tf.random_normal([128])),
    'bc3_1': tf.Variable(tf.random_normal([256])),
    'bc3_2': tf.Variable(tf.random_normal([256])),
    'bc3_3': tf.Variable(tf.random_normal([256])),
    'bc4_1': tf.Variable(tf.random_normal([512])),
    'bc4_2': tf.Variable(tf.random_normal([512])),
    'bc4_3': tf.Variable(tf.random_normal([512])),
    'bc5_1': tf.Variable(tf.random_normal([512])),
    'bc5_2': tf.Variable(tf.random_normal([512])),
    'bc5_3': tf.Variable(tf.random_normal([512])),
    'bc6': tf.Variable(tf.random_normal([1800])),
    'bc7': tf.Variable(tf.random_normal([300])),
    'bc8': tf.Variable(tf.random_normal([60])),
    'bc9': tf.Variable(tf.random_normal([4]))
}

#tf Graph Input
X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_LAYER])
Y = tf.placeholder(tf.float32, [None, CLASS_LABEL])
keep_prop = tf.placeholder(tf.float32)

# Construct model
pred = conv_net(X, weights, biases, keep_prop)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

cost_sum = tf.summary.scalar("cost", cost)


# Evaluate model
is_correct = tf.equal(tf.arg_max(pred, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

acc_sum = tf.summary.scalar("accuracy", accuracy)


th = threading.Thread(target=get_data, args=(data,))
th.daemon = True
th.start()


with tf.Session() as sess:

    summary = tf.summary.merge_all()

    writer = tf.summary.FileWriter('./logs/rate3_100_VGG')
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    oneBatchNum = int(training_Set / BATCH_SIZE)
    for epoch in range(TRAINING_EPOCHS):
        avg_cost = 0

        for i in range(BATCH_SIZE):
            batch_ys = oneHotEncoding(trainingScore[oneBatchNum * i:oneBatchNum * (i + 1)])
            batch_xs = ImageData(trainingName[oneBatchNum * i:oneBatchNum * (i + 1)])

            print(i)

            c, p, s, _ = sess.run([cost, pred, summary, train], feed_dict={X: batch_xs, Y: batch_ys, keep_prop: 0.7})
            avg_cost += c / BATCH_SIZE



        print('Epoch:', "%d" % (epoch + 1), 'cost = ', '{:.6f}', format(avg_cost))

    WC1_1 = weights['wc1_1'].eval(sess)
    WC1_2 = weights['wc1_2'].eval(sess)
    WC2_1 = weights['wc2_1'].eval(sess)
    WC2_2 = weights['wc2_2'].eval(sess)
    WC3_1 = weights['wc3_1'].eval(sess)
    WC3_2 = weights['wc3_2'].eval(sess)
    WC3_3 = weights['wc3_3'].eval(sess)
    WC4_1 = weights['wc4_1'].eval(sess)
    WC4_2 = weights['wc4_2'].eval(sess)
    WC4_3 = weights['wc4_3'].eval(sess)
    WC5_1 = weights['wc5_1'].eval(sess)
    WC5_2 = weights['wc5_2'].eval(sess)
    WC5_3 = weights['wc5_3'].eval(sess)
    WC6 = weights['wc6'].eval(sess)
    WC7 = weights['wc7'].eval(sess)
    WC8 = weights['wc8'].eval(sess)
    WC9 = weights['wc9'].eval(sess)

    BC1_1 = biases['bc1_1'].eval(sess)
    BC1_2 = biases['bc1_2'].eval(sess)
    BC2_1 = biases['bc2_1'].eval(sess)
    BC2_2 = biases['bc2_2'].eval(sess)
    BC3_1 = biases['bc3_1'].eval(sess)
    BC3_2 = biases['bc3_2'].eval(sess)
    BC3_3 = biases['bc3_3'].eval(sess)
    BC4_1 = biases['bc4_1'].eval(sess)
    BC4_2 = biases['bc4_2'].eval(sess)
    BC4_3 = biases['bc4_3'].eval(sess)
    BC5_1 = biases['bc5_1'].eval(sess)
    BC5_2 = biases['bc5_2'].eval(sess)
    BC5_3 = biases['bc5_3'].eval(sess)
    BC6 = biases['bc6'].eval(sess)
    BC7 = biases['bc7'].eval(sess)
    BC8 = biases['bc8'].eval(sess)
    BC9 = biases['bc9'].eval(sess)

g = tf.Graph()
with g.as_default():
    X_2 = tf.placeholder("float", [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_LAYER], name="input")
    x_image = tf.reshape(X_2, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_LAYER])

    WC1_1 = tf.constant(WC1_1, name="WC1_1")
    WC1_2 = tf.constant(WC1_2, name="WC1_2")
    WC2_1 = tf.constant(WC2_1, name="WC2_1")
    WC2_2 = tf.constant(WC2_2, name="WC2_2")
    WC3_1 = tf.constant(WC3_1, name="WC3_1")
    WC3_2 = tf.constant(WC3_2, name="WC3_2")
    WC3_3 = tf.constant(WC3_3, name="WC3_3")
    WC4_1 = tf.constant(WC4_1, name="WC4_1")
    WC4_2 = tf.constant(WC4_2, name="WC4_2")
    WC4_3 = tf.constant(WC4_3, name="WC4_3")
    WC5_1 = tf.constant(WC5_1, name="WC5_1")
    WC5_2 = tf.constant(WC5_2, name="WC5_2")
    WC5_3 = tf.constant(WC5_3, name="WC5_3")
    WC6 = tf.constant(WC6, name="WC6")
    WC7 = tf.constant(WC7, name="WC7")
    WC8 = tf.constant(WC8, name="WC8")
    WC9 = tf.constant(WC9, name="WC9")
    BC1_1 = tf.constant(BC1_1, name="BC1_1")
    BC1_2 = tf.constant(BC1_2, name="BC1_2")
    BC2_1 = tf.constant(BC2_1, name="BC2_1")
    BC2_2 = tf.constant(BC2_2, name="BC2_2")
    BC3_1 = tf.constant(BC3_1, name="BC3_1")
    BC3_2 = tf.constant(BC3_2, name="BC3_2")
    BC3_3 = tf.constant(BC3_3, name="BC3_3")
    BC4_1 = tf.constant(BC4_1, name="BC4_1")
    BC4_2 = tf.constant(BC4_2, name="BC4_2")
    BC4_3 = tf.constant(BC4_3, name="BC4_3")
    BC5_1 = tf.constant(BC5_1, name="BC5_1")
    BC5_2 = tf.constant(BC5_2, name="BC5_2")
    BC5_3 = tf.constant(BC5_3, name="BC5_3")
    BC6 = tf.constant(BC6, name="BC6")
    BC7 = tf.constant(BC7, name="BC7")
    BC8 = tf.constant(BC8, name="BC8")
    BC9 = tf.constant(BC9, name="BC9")

    CONV1_1 = conv2d(x_image, WC1_1, BC1_1)
    CONV1_2 = conv2d(CONV1_1, WC1_2, BC1_2)
    POOL1 = maxpool2d(CONV1_2)

    CONV2_1 = conv2d(POOL1, WC2_1, BC2_1)
    CONV2_2 = conv2d(CONV2_1, WC2_2, BC2_2)
    POOL2 = maxpool2d(CONV2_2)

    CONV3_1 = conv2d(POOL2, WC3_1, BC3_1)
    CONV3_2 = conv2d(CONV3_1, WC3_2, BC3_2)
    CONV3_3 = conv2d(CONV3_2, WC3_3, BC3_3)
    POOL3 = maxpool2d(CONV3_2)

    CONV4_1 = conv2d(POOL3, WC4_1, BC4_1)
    CONV4_2 = conv2d(CONV4_1, WC4_2, BC4_2)
    CONV4_3 = conv2d(CONV4_2, WC4_3, BC4_3)
    POOL4 = maxpool2d(CONV4_3)

    CONV5_1 = conv2d(POOL4, WC5_1, BC5_1)
    CONV5_2 = conv2d(CONV5_1, WC5_2, BC5_2)
    CONV5_3 = conv2d(CONV5_2, WC5_3, BC5_3, strid=2)
    POOL5 = maxpool2d(CONV5_3)

    FC1 = tf.reshape(POOL5, [-1, 8192])
    FC1 = tf.nn.relu(tf.matmul(FC1, WC6) + BC6)

    FC2 = tf.nn.relu(tf.matmul(FC1, WC7) + BC7)

    FC3 = tf.nn.relu(tf.matmul(FC2, WC8) + BC8)

    FC4 = tf.add(tf.matmul(FC3, WC9), BC9)

    softMaxTest = tf.nn.softmax(FC4, name="output")

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    y_train = tf.placeholder("float", [None, CLASS_LABEL])
    correct_prediction = tf.equal(tf.argmax(FC4, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    tf.train.write_graph(g.as_graph_def(), 'models/', 'rate3_100_text_VGG_6000.pb', as_text=True)
    tf.train.write_graph(g.as_graph_def(), 'models/', 'rate3_100_VGG_6000.pb', as_text=False)



    i = 0

    result = open("RESULT_6000_3.txt", 'w')


    while True:
        if len(testName) < (i + 1) * 20:
            break
        input_y = oneHotEncoding(testScore[i * 20:(i + 1) * 20])
        result.write("accuracy %g" % accuracy.eval(
        {X_2: ImageData(testName[i * 20:(i + 1) * 20]), y_train: input_y}, sess))

        result.write("\n")
        SoftMax_Result = softMaxTest.eval({X_2: ImageData(testName[i * 20:(i + 1) * 20])}, sess)

        for j in range(len(SoftMax_Result)):
            result.write(str(SoftMax_Result[j]) + "\n" + str(input_y[j]) + "\n")

    i += 1