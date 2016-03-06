from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from data import create_example
from ham import HAMOperations, HAMNode, HAMTree

class Config(object):
  n = 4
  embed_size = 10
  tree_size = 20
  controller_size = 20

config = Config()

inputs = tf.placeholder(tf.float32, shape=[config.n, config.embed_size], name='Input')
control = tf.placeholder(tf.float32, shape=[1, config.controller_size], name='Control')
target = tf.placeholder(tf.float32, shape=[config.embed_size], name='Target')

ham_ops = HAMOperations(config.embed_size, config.tree_size, config.controller_size)
tree = HAMTree(ham_ops=ham_ops)
tree.construct(config.n)

values = tf.split(0, config.n, inputs)
for i, val in enumerate(values):
  tree.leaves[i].embed(val)
tree.refresh()

calculate_predicted = tree.get_output(control)
calculate_loss = tf.reduce_sum(tf.pow(calculate_predicted - target, 2))

optimizer = tf.train.AdamOptimizer(0.001)
train_step = optimizer.minimize(calculate_loss)

init = tf.initialize_all_variables()
saver = tf.train.Saver()
saved_weights_fn = './ham.weights'

with tf.Session() as session:
  session.run(init)

  if os.path.exists(saved_weights_fn):
    saver.restore(session, saved_weights_fn)

  epoch = 0
  while 1:
    epoch += 1
    total_examples = 1000
    total_accuracy = 0.0
    total_loss = 0.0
    for i in xrange(total_examples):
      X, Y = create_example(n=config.n, bit_length=config.embed_size / 2)
      control_signal = np.zeros([1, config.controller_size], dtype=np.float32)
      feed = {inputs: X, target: Y[0], control: control_signal}
      _, loss, predicted = session.run([train_step, calculate_loss, calculate_predicted], feed_dict=feed)
      ###
      y = Y[0].astype(int)
      y_pred = np.rint(predicted).astype(int)[0]
      #print('True = {}'.format(y))
      #print('Pred = {}'.format(y_pred))
      total_accuracy += (y == y_pred).all()
      total_loss += loss
    print('Epoch = {}'.format(epoch))
    print('Loss = {}'.format(total_loss / total_examples))
    print('Accuracy = {}'.format(total_accuracy / total_examples))
    print('=-=')
    saver.save(session, saved_weights_fn)