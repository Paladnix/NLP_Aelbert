import matplotlib.pyplot as plt
import numpy as np


def read_data(file_name):
   
    data = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            data.append(float(lines[i]))
        return data




def draw_roc():
  model_A = "Bert"
  model_B = "Ours"
  # plot them
  A_fpr = read_data("./Bert-paws.fpr")
  B_fpr = read_data("./Ours.fpr")
  A_tpr = read_data("./Bert-paws.tpr")
  B_tpr = read_data("./Ours.tpr")

  plt.plot(A_fpr,A_tpr,linestyle='-', label='BERT', color='orange')
  plt.plot(B_fpr,B_tpr,linestyle='-', label='Ours', color='dodgerblue')
  plt.legend()

  plt.savefig("./paws-roc.png")


if __name__ == '__main__':
    draw_roc()
