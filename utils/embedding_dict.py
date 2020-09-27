#!/usr/bin/env python
# coding=utf-8

import numpy as np

class EmbeddingDict:

  def __init__(self, dict_file, words=None, split=" ", size=-1):
    """
    Args:
      split: 字典的分隔符
    """
    self.dict_file = dict_file
    self.words = words
    self.split = split
    self.size = size
    if size != -1:
      self.default_embedding = np.zeros(self.size)
      self.dict = defaultdict(self.default_embedding) 
    self.load_dict()


  def load_dict(self):
    with open(self.dict_file) as f:
      while True:
        line = f.readline()
        if line is None:
          break
        splits = line.split(self.split)
        word = splits[0]
        if self.size == -1:
          self.size = len(splits) - 1
          self.default_embedding = np.zeros(self.size)
          self.dict = defaultdict(self.default_embedding) 
        embedding = np.array([float(s) for s in splits[1:]])
        embedding_dict[word] = embedding

  def get_word_embedding(self, word):

    return self.dict[char]


  # TODO padding到固定长度  
  def get_sentence_embedding(self, sentence):
    return [ self.dict[word] for word in sentence ]
