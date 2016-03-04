from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Transformer(object):
  '''
  Input = [b]
  Output = [d]
  '''
  def __init__(self, input_size, output_size):
    self.W = tf.Variable(tf.truncated_normal([input_size, output_size]))

  def __call__(self, left, right):
    return tf.nn.relu(tf.matmul(tf.concat(1, [left, right]), self.W))


class Join(Transformer):
  '''
  Input = [d, d]
  Output = [d]
  '''
  def __init__(self, d):
    self.W = tf.Variable(tf.truncated_normal([2 * d, d]))

  def __call__(self, left, right):
    return tf.nn.relu(tf.matmul(tf.concat(1, [left, right]), self.W))


class Search(Transformer):
  '''
  Input = [d, l]
  Output = [1]
  '''
  def __init__(self, d, l):
    self.W = tf.Variable(tf.truncated_normal([d + l, 1]))

  def __call__(self, h, control):
    return tf.nn.sigmoid(tf.matmul(tf.concat(1, [h, control]), self.W))


class Write(Transformer):
  '''
  Input = [d, l]
  Output = [d]
  '''
  def __init__(self, d, l):
    self.H = tf.Variable(tf.truncated_normal([d + l, d]))
    self.T = tf.Variable(tf.truncated_normal([d + l, 1]))

  def __call__(self, h, control):
    data = tf.concat(1, [h, control])
    candidate = tf.nn.sigmoid(tf.matmul(data, self.H))
    update = tf.nn.sigmoid(tf.matmul(data,  self.T))
    return update * candidate + (1 - update) * h


class HAMTree(object):
  def __init__(self, embed_size, tree_size, controller_size):
    ###
    # From the paper,
    self.embed_size = embed_size  # b
    self.tree_size = tree_size  # d
    self.controller_size = controller_size  # l
    ##
    self.transform = Transformer(embed_size, tree_size)
    self.join = Join(tree_size)
    self.search = Search(tree_size, controller_size)
    self.write = Write(tree_size, controller_size)
    ##
    self.root = None
    self.leaves = None

  def construct(self, depth):
    # Create all the leaf nodes, then combine them
    # A B C D
    # C D [A B]
    # [A B] [C D]
    # [[A B] [C D]]
    ###
    total_leaves = 2 ** depth
    stack = [HAMNode(tree=self, left=None, right=None) for leaf in xrange(total_leaves)]
    self.leaves = stack
    while len(stack) > 1:
      l, r = stack[:2]
      stack = stack[2:]
      stack.append(HAMNode(tree=self, left=l, right=r))
    self.root = stack[0]

  def get_output(self, control):
    return self.root.get_output(control)


class HAMNode(object):
  def __init__(self, tree, left, right):
    self.tree = tree
    self.left = left
    self.right = right
    self.h = tree.join(left, right)

  def embed(self, value):
    self.h = self.tree.transform(value)

  def get_output(self, control):
    value = None
    ###
    # Retrieve the value - left and right weighted by the value of search
    if self.left and self.right:
      decision = self.tree.search(self.h, control)
      value = decision * self.right.get_output(control)
      value += (1 - decision) * self.left.get_output(control)
    else:
      value = self.h
    ###
    # Update the values of the tree
    if self.left and self.right:
      self.h = self.tree.join(self.left.h, self.right.h)
    else:
      self.h = self.tree.write(self.h, control)
    return value