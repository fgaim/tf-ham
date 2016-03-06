from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

ham_ops = HAMOperations(config.embed_size, config.tree_size, config.controller_size)
tree = HAMTree(ham_ops=ham_ops)
tree.construct(config.n)

X, Y = create_example(n=config.n, bit_length=config.embed_size / 2)
print(X)
for i, val in enumerate(X):
  val = np.array([val])
  print('Embedding into leaf {}'.format(i))
  tree.leaves[i].embed(val)
tree.refresh()

init = tf.initialize_all_variables()

with tf.Session() as session:
  session.run(init)

  print(session.run(tree.root.h))
  control = np.zeros([1, config.controller_size], dtype=np.float32)
  print(session.run(tree.get_output(control)))
