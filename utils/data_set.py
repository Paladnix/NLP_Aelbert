#!/usr/bin/env python
# coding=utf-8

import data_utils as utils

class Data_Set:
  """ 数据集基类

  """

  def __init__(self, data_file):
    self.data_file = data_file
    self.get_words()


  def get_line_text(self):
    """ 获取一条测试数据原始文本    
    Args： 
      None

    Return:
      line iterator
    """
    # TODO 可以针对内存有优化空间
    with open(self.data_file) as f:
      for line in f.readlines():
        yield line


  def get_words(self):
    if self.words is not None:
      return self.words
    self.words = set()
    with open(self.data_file) as f:
      while True:
        line = f.readline()
        if line is None:
          break
        for word in self.text_2_sequence(line):
          self.words.add(word)
    
    return self.words
        

  def text_2_sequence(self, text):
    # TODO 特殊符号是否要剥离出来
    return text.split()
