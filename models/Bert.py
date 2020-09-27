import torch 
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForPreTraining
import torch.nn.functional as F
import logging

import os
current_path = os.path.dirname(__file__)
BERT_MODEL_PATH = current_path + "/../../_model/bert-base/"

import time, datetime
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
from torch.autograd import Variable
import numpy as np
import gc


class Bert(nn.Module):

    def __init__(self, n_classes=2):
        super(Bert, self).__init__()
        self.model_name = 'BertModel'
        self.n_classes = n_classes
        self.bert_model_1 = BertModel.from_pretrained('bert-base-uncased')
        
        self.classifer = nn.Sequential(nn.Linear(768, self.n_classes))

        self.loss_fct = nn.CrossEntropyLoss()



    def forward(self, ids, labels):

        # print("Input Size: ",  ids.size(), labels.size())

        labels = labels.type(torch.FloatTensor).cuda()
        outputs = self.bert_model_1(ids)

        # print("Bert Outputs Size: ",  outputs_1[1].size(), outputs_2[1].size())

        outs = self.classifer(outputs[1])
        
        # print("Classifer Outputs Size: ", outs_1.size(), outs_2.size(), outs.size())

        loss = self.loss_fct(outs.view(-1, self.n_classes), labels.type(torch.LongTensor).view(-1).cuda())
	    
        outputs = torch.max(F.softmax(outs, dim=1), dim=1)

        #print("Outputs: ", outputs)
        #print("Loss: ", loss.item())
        #print("Loss_1: ", loss_1.size())
        #print("Loss_2: ", loss_2.size())

        return outputs, loss


def load_model(reload_model_name=None, CUDA=False, GPUs=None):
    # 加载模型
    model = Bert()
    if CUDA:
        model = model.cuda()
        if GPUs is not None and len(GPUs) > 0:
            model = torch.nn.DataParallel(model, device_ids=GPUs)
    if reload_model_name is not None:
        logging.info("load model from: %s" % reload_model_name)
        model.load_state_dict(torch.load(reload_model_name))

    return model


def save_ROC(y, prob, filename):
    fpr, tpr, thresholds = roc_curve(y, prob)

    with open(filename + ".fpr", 'w') as f:
        for x in fpr:
            f.write(str(x) + "\n")
    with open(filename + ".tpr", 'w') as f:
        for x in tpr:
            f.write(str(x) + "\n")


        #xnew = np.linspace(fpr.min(),fpr.max(),300)
        #ynew = make_interp_spline(fpr,tpr)(xnew)
        #plt.plot(xnew,ynew)
        #plt.plot(fpr,tpr,'-')
        #plt.savefig("./roc.png")



def metric_fn(p, t):
    # p=(p>0.5)*1
    return accuracy_score(t, p), roc_auc_score(t, p), f1_score(t,p)


def validation_fn(data_loader, model, config):
    dev_data = data_loader.get_dev(config.dev_data)  
    model.eval()
    y_score, y_pred, y_true, tloss = [], [], [], []
    with torch.no_grad():
        for i, (ids, seg_ids, target, rows) in enumerate(tqdm(dev_data)):
            outputs, loss = model(ids.cuda(), target.cuda())
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
        acc, auc, f1 = metric_fn(y_pred, y_true, )
        save_ROC(y_true, np.array(y_score), config.log_dir + "/"+ config.model_id + datetime.datetime.now().strftime("_%m-%d_%H_%M"))
        logging.info("DEV")
        logging.info("tloss: %.4f" % tloss)
        logging.info("acc:   %.4f" % acc)
        logging.info("auc:   %.4f" % auc)
        logging.info("f1:    %.4f" % f1)
    
        return tloss, acc, auc, f1, y_true, y_pred, y_score


def train(data_loader, model, config):

    trainning_data = data_loader.get_train(config.train_data)
    # 训练选取的优化器，由训练部分决定
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    # 对数据正向计算，获取输出结果
    for epoch in range(config.num_epoch):
        logging.info("The %d epoch.", epoch)
        y_pred, y_true, tloss = [], [], []
        model.train()
        for i, (ids, seg_ids, labels, rows) in enumerate(tqdm(trainning_data)):
            if ids is None:
                continue
            outputs, loss = model(Variable(ids.cuda()), Variable(labels.cuda()))
            tloss.append(loss.mean().item())
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

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

        validation_fn(data_loader, model, config)

        torch.save(model.state_dict(), config.model_save_dir + "/" + config.model_id + datetime.datetime.now().strftime("_%m-%d_%H_%M")+"-"+str(epoch))
