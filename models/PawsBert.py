import torch 
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForPreTraining
import torch.nn.functional as F

import os
current_path = os.path.dirname(__file__)
BERT_MODEL_PATH = current_path + "/../../_model/bert-base/"

class PawsBertModel(nn.Module):

    def __init__(self, n_classes=2):
        super(PawsBertModel, self).__init__()
        self.model_name = 'PawsBertModel'
        self.n_classes = n_classes
        self.bert_model_1 = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model_2 = BertModel.from_pretrained('bert-base-uncased')
        
        self.classifer_pre_1 = nn.Sequential(nn.Linear(768, self.n_classes))
        self.classifer_pre_2 = nn.Sequential(nn.Linear(768, self.n_classes))
        self.classifer_last = nn.Sequential(nn.Linear(768*2, self.n_classes))

        self.loss_fct = nn.CrossEntropyLoss()



    def forward(self, ids, labels):

        # print("Input Size: ",  ids.size(), labels.size())

        labels = labels.type(torch.FloatTensor).cuda()
        outputs_1 = self.bert_model_1(ids)
        outputs_2 = self.bert_model_2(ids.clone())

        # print("Bert Outputs Size: ",  outputs_1[1].size(), outputs_2[1].size())

        outs_1 = self.classifer_pre_1(outputs_1[1])
        outs_2 = self.classifer_pre_2(outputs_2[1])
        def diff_loss(s, t):
            s = s - torch.mean(s, 0)
            t = t - torch.mean(t, 0)

            s = F.normalize(s, dim=0, p=2)
            t = F.normalize(t, dim=0, p=2)

            correlation_matrix = torch.matmul(torch.t(s),t)
            cost = torch.mean(correlation_matrix**2)*0.01

            cost = torch.where(cost>0, cost, torch.tensor(0).float().cuda())

            return cost
        cor_loss_1 = diff_loss(outputs_1[1], outputs_2[1].detach())
        cor_loss_2 = diff_loss(outputs_1[1].detach(), outputs_2[1])

        outs = self.classifer_last(torch.cat([outputs_1[1], outputs_2[1]], dim=1).detach()) 
        
        # print("Classifer Outputs Size: ", outs_1.size(), outs_2.size(), outs.size())

        ones = torch.ones_like(labels).type(torch.FloatTensor).cuda()
        nega_labels = (ones - labels).type(torch.LongTensor).cuda()

        # print("Nega_labels: ", nega_labels)
        # print("Labels: ", labels.type(torch.LongTensor))
        
        loss_1 = self.loss_fct(outs_1.view(-1, self.n_classes), labels.type(torch.LongTensor).view(-1).cuda()) + cor_loss_1
        loss_2 = self.loss_fct(outs_2.view(-1, self.n_classes), nega_labels.view(-1)) + cor_loss_2
        
        loss = self.loss_fct(outs.view(-1, self.n_classes), labels.type(torch.LongTensor).view(-1).cuda())
	    
        # 输出预测标签。
        outs = F.softmax(outs, dim=1)
        outputs = torch.max(outs, dim=1)
        # outputs = F.softmax(outs, dim=1)
        #print("Outputs: ", outputs)
        #print("Loss: ", loss.item())
        #print("Loss_1: ", loss_1.size())
        #print("Loss_2: ", loss_2.size())

        return outputs, loss, loss_1, loss_2


if __name__ == '__main__':
    # nega_labels = nega_labels.type(torch.LongTensor)
    # print(nega_labels)
    model = PawsBertModel()
    print(model)

    parameters = list(model.named_parameters())
    for para in parameters:
        print(para)



