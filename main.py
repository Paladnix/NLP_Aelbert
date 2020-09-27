# coding=utf-8
import torch
import sys
import time, datetime
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
from torch.autograd import Variable
import numpy as np
import gc
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

config = None 

import logging
sys.path.insert(0, './utils')
import config_parser, PawsDataSet
sys.path.insert(1, './models')
from PawsBert import PawsBertModel

MODEL_SAVE_DIR = './_model'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
gpu_ids = [0,1,2]
CUDA = False
if torch.cuda.is_available():
    CUDA = True

def draw_ROC(y, prob):
    fpr, tpr, thresholds = roc_curve(y, prob)

    with open("./plot/Ours.fpr", 'w') as f:
        for x in fpr:
            f.write(str(x) + "\n")
    with open("./plot/Ours.tpr", 'w') as f:
        for x in tpr:
            f.write(str(x) + "\n")


    #xnew = np.linspace(fpr.min(),fpr.max(),300)
    #ynew = make_interp_spline(fpr,tpr)(xnew)
    #plt.plot(xnew,ynew)
    #plt.plot(fpr,tpr,'-')
    #plt.savefig("./roc.png")

def get_wrong_case(rows, outputs, labels):
    outputs = (outputs > 0.5) * 1
    for i in range(len(outputs)):
        if outputs[i] != labels[i]:
            print(rows[i])	
            print(labels[i].item())


def metric_fn(p, t):
    # p=(p>0.5)*1
    return accuracy_score(t, p), roc_auc_score(t, p), f1_score(t,p)

def validation_fn(model):

    test_loader = PawsDataSet.get_test_loader(config)
    model.eval()
    y_score, y_pred, y_true, tloss = [], [], [], []
    with torch.no_grad():
        for i, (ids, seg_ids, target, rows) in enumerate(tqdm(test_loader)):
            outputs, loss, loss_1, loss_2 = model(ids.cuda(), target.cuda())
            # get_wrong_case(rows, outputs.squeeze(), target)
            # loss = loss_fn(outputs.squeeze().cuda(), target.type(torch.FloatTensor).cuda())
            print(outputs)
            tloss.append(loss.mean().item())
            y_true.append(target.detach().cpu().numpy())
            y_pred.append(outputs[1].detach().cpu().numpy())
            for i in range(len(outputs[0])):
                if outputs[1][i].item() == 1:
                    y_score.append(outputs[0][i].item())
                else :
                    y_score.append(1 - outputs[0][i].item())
                                                                
    tloss = np.array(tloss).mean()
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    acc, auc, f1 = metric_fn(y_pred, y_true)
    draw_ROC(y_true, np.array(y_score))
    logging.info("DEV_AND_TEST")
    logging.info("tloss: %.4f" % tloss)
    logging.info("acc:   %.4f" % acc)
    logging.info("auc:   %.4f" % auc)
    logging.info("f1:    %.4f" % f1)
    return tloss, acc, auc, f1


def _train():
    
    data_loader = PawsDataSet.get_train_loader(config)
    # 加载模型
    model = PawsBertModel()
    if CUDA:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    if config.load and config.reload_model_name is not None:
        logging.info("load model from: %s" % config.reload_model_name)
        model.load_state_dict(torch.load(config.reload_model_name))

    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    # 对数据正向计算，获取输出结果
    for epoch in range(config.num_epoch):

        logging.info("The %d epoch.", epoch)
        y_pred, y_true, tloss = [], [], []
        model.train()
        for i, (ids, seg_ids, labels, rows) in enumerate(tqdm(data_loader)):
            if ids is None:
                print("ids is None")
                continue
            outputs, loss, loss_1, loss_2 = model(Variable(ids.cuda()), Variable(labels.cuda()))
            # print(outputs, loss, loss_1, loss_2)
            # loss = loss_fn(outputs.squeeze().cuda(), Variable(labels.type(torch.FloatTensor).cuda()))
            tloss.append(loss.mean().item())
            optimizer.zero_grad()
            loss.mean().backward()
            loss_1.mean().backward()
            loss_2.mean().backward()
            optimizer.step()
#            print(loss.item())
#            for name, parms in model.named_parameters():
#                print('-->name:', name, '\n-->grad_requirs:',parms.requires_grad, \
#                      '\n-->grad_value:',parms.grad)
            y_true.append(labels.detach().cpu().numpy())
            y_pred.append(outputs[1].detach().cpu().numpy())
            y_t = np.concatenate(y_pred)
            y_p = np.concatenate(y_true)
            del ids, seg_ids, labels
            gc.collect()

        tloss = np.array(tloss).mean()
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        acc, auc, f1 = metric_fn(y_pred, y_true)
        logging.info("Train")
        logging.info("tloss: %.4f" % tloss)
        logging.info("acc:   %.4f" % acc)
        logging.info("auc:   %.4f" % auc)
        logging.info("f1:    %.4f" % f1)
        dev_tloss, dev_acc, dev_auc, dev_f1 = validation_fn(model)
    
        torch.save(model.state_dict(), MODEL_SAVE_DIR + "/" + config.model_id + '-' + datetime.datetime.now().strftime("_%m-%d_%H_%M"))
        # if dev_acc > 0.92:
        #    break


def _dev():
    # 加载模型
    model = PawsBertModel()
    if CUDA:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    if config.load and config.reload_model_name is not None:
        logging.info("load model from: %s" % config.reload_model_name)
        model.load_state_dict(torch.load(config.reload_model_name))

    dev_tloss, dev_acc, dev_auc, dev_f1 = validation_fn(model)
    



def _test():
    
    model = PawsBertModel()
    if config.load and config.reload_model_name is not None:
        logging.info("load model from: %s" % config.reload_model_name)
        model.load_state_dict(torch.load(config.reload_model_name))
    if CUDA:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    test_loader = PawsDataSet.get_test_loader(config)
    model.eval()
    for i, (ids, seg_ids, label, rows) in enumerate(tqdm(test_loader)):
        outputs = model(ids.cuda(), seg_ids.cuda())
        # get_wrong_case(rows, outputs.squeeze(), target)
        # loss = loss_fn(outputs.squeeze().cuda(), target.type(torch.FloatTensor).cuda())
        tloss.append(loss.item())
        y_true.append(target.detach().cpu().numpy())
        y_pred.append(outputs.squeeze().detach().cpu().numpy())
                                                                
    tloss = np.array(tloss).mean()
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    acc, auc, f1 = metric_fn(y_pred, y_true)
    logging.info("DEV_AND_TEST")
    logging.info("tloss: %.4f" % tloss)
    logging.info("acc:   %.4f" % acc)
    logging.info("auc:   %.4f" % auc)
    logging.info("f1:    %.4f" % f1)
    return tloss, acc, auc, f1
    


if __name__ == '__main__':
  config = config_parser.parser.parse_args()
  logging.basicConfig(
    filename="./runtime/"+ config.model_id + datetime.datetime.now().strftime("_%m-%d_%H") + config.mode + '.log',
    level=logging.INFO, format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
  logging.info(config)
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  torch.cuda.manual_seed(config.seed)

  if config.mode == 'train':
    _train()
  elif config.mode == 'test':
    _test()
  elif config.mode == 'dev':
    _dev()
  else:
    raise ValueError("invalid value for 'mode': {}".format(config.mode))
