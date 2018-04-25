import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LeNet_inference

import numpy as np


BATCH_SIZE=100
LEARNING_RATE_BASE=0.01
LEARNING_RATE_DEACY=0.99
REGULARAZTION_RATE=0.0001
TRAINNNNING_STEPS=50000
MOVING_AVERAGE_DECAY=0.99
MODEL_SAVE_PATH="/home/sam/Documents/Python_work/LearningPython/tensorflow/MODEL_SAVE4"
MODEL_NAME="model.ckpt"

def train(mnist):

    x=tf.placeholder(tf.float32,[BATCH_SIZE,LeNet_inference.IMAGE_SIZE,LeNet_inference.IMAGE_SIZE,LeNet_inference.NUM_CHANNELS],name='x-input')

    y_=tf.placeholder(tf.float32,[None,LeNet_inference.OUTPUT_NODE],name='y-input')

    regularizer=tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    y=LeNet_inference.inference(x,True,regularizer)
    global_step=tf.Variable(0,trainable=False)


    varialbe_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_average_op=varialbe_averages.apply(tf.trainable_variables())

    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)

    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DEACY)

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    with tf.control_dependencies([train_step,variable_average_op]):
        train_op=tf.no_op(name='train')

    saver=tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINNNNING_STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs=np.reshape(xs,(BATCH_SIZE,LeNet_inference.IMAGE_SIZE,LeNet_inference.IMAGE_SIZE,LeNet_inference.NUM_CHANNELS))
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})
            if i%1000==0:
                print("After %d training steps,loss on training batch is %g."%(step,loss_value))
                saver.save(sess,"/home/sam/Documents/Python_work/LearningPython/tensorflow/MODEL_SAVE4/model.ckpt",global_step=global_step)


def main(argv=None):
    mnist=input_data.read_data_sets("/home/sam/Documents/Python_work/LearningPython/tensorflow/tensorflow_train_data",one_hot=True)
    train(mnist)

if __name__=='__main__':
    tf.app.run()

