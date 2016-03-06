from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from data import create_example
from ham import HAMOperations, HAMNode, HAMTree

class Config(object):
  max_epochs = 100
  batch_size = 50
  n = 8
  embed_size = 10
  tree_size = 20
  controller_size = 20

config = Config()

inputs = tf.placeholder(tf.float32, shape=[config.batch_size, config.n, config.embed_size], name='Input')
control = tf.placeholder(tf.float32, shape=[config.batch_size, config.controller_size], name='Control')
target = tf.placeholder(tf.float32, shape=[config.batch_size, config.embed_size], name='Target')

ham_ops = HAMOperations(config.embed_size, config.tree_size, config.controller_size)
tree = HAMTree(ham_ops=ham_ops)
tree.construct(config.n)

values = [tf.squeeze(x, [1]) for x in tf.split(1, config.n, inputs)]
for i, val in enumerate(values):
  tree.leaves[i].embed(val)
tree.refresh()

calculate_predicted = tree.get_output(control)
calculate_loss = tf.reduce_sum(tf.pow(calculate_predicted - target, 2)) / config.batch_size

optimizer = tf.train.AdamOptimizer(0.001)
train_step = optimizer.minimize(calculate_loss)

init = tf.initialize_all_variables()
saver = tf.train.Saver()
saved_weights_fn = './ham.weights'

with tf.Session() as session:
  session.run(init)

  if os.path.exists(saved_weights_fn):
    saver.restore(session, saved_weights_fn)

  # Paper uses 100 epochs with 1000 batches of batch size 50
  for epoch in xrange(config.max_epochs):
    total_batches = 1000
    total_accuracy = 0.0
    total_loss = 0.0
    for i in xrange(total_batches):
      X, Y = [], []
      for b in xrange(config.batch_size):
        x, y = create_example(n=config.n, bit_length=config.embed_size / 2)
        X.append(x)
        Y.append(y[0])
      control_signal = np.zeros([config.batch_size, config.controller_size], dtype=np.float32)
      feed = {inputs: X, target: Y, control: control_signal}
      _, loss, predicted = session.run([train_step, calculate_loss, calculate_predicted], feed_dict=feed)
      ###
      for y, y_pred in zip(Y, predicted):
        y = y.astype(int)
        y_pred = np.rint(y_pred).astype(int)
        total_accuracy += (y == y_pred).all()
      total_loss += loss
    print('Epoch = {}'.format(epoch))
    print('Loss = {}'.format(total_loss / total_batches))
    print('Accuracy = {}'.format(total_accuracy / (config.batch_size * total_batches)))
    print('=-=')
    saver.save(session, saved_weights_fn)