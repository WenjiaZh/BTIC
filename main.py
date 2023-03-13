#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:33:52 2021

@author: wjz
"""
import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
import torchvision.models as models
from transformers import  BertTokenizer, BertModel
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns



imagepath = 'load images from'
datapath = 'load data from'
savepath = 'save to'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

##############################
### split and embed images ###
##############################

resnet = models.resnet50(pretrained=True)
resnet50 = nn.Sequential(*list(resnet.children())[:-1])   

for param in resnet50.parameters():
    param.requires_grad = False

def cut_image(image):
    width, height = image.size
    item_width = int(width / 8)
    box_list = []    
    # (left, upper, right, lower)
    for i in range(0,8):
        for j in range(0,8):           
            box = (j*item_width,i*item_width,(j+1)*item_width,(i+1)*item_width)
            box_list.append(box)
        image_list = [np.array(image.crop(box)) for box in box_list]    
    return image_list


filename = []
img_64 = []
for fileName in os.listdir(imagepath):
    if os.path.splitext(fileName)[1] == '.jpg':
        filesname = os.path.splitext(fileName)[0]
        filename.append(filesname)
        filepath = imagepath+'/'+filesname+'.jpg'
        image = Image.open(filepath).convert('RGB')
        image = image.resize((640, 640),Image.ANTIALIAS)
        image_list = cut_image(image)
        images = torch.tensor(image_list)
        images = images.permute(0,3,1,2).float()
        images = resnet50(images)
        images = images.squeeze(3).squeeze(2).numpy()
        img_64.append(images)
        print(filesname)

image_64 = pd.DataFrame({'Id':filename,'imageembd':img_64}) 

#save and load embeded image patches sequences
#np.save(savepath+'/image_64.npy',image_64)
#image = np.load(datapath+'/image_64.npy', allow_pickle=True)
#image = pd.DataFrame(image,columns=['Id', 'img'])
image = pd.DataFrame(image_64,columns=['Id', 'img'])
image.Id = image.Id.astype(int)



#################
### load text ###
#################
text = pd.read_csv(datapath+'/text.csv')

#Field
label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
id_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.int)
fields = [('Id',id_field),('label', label_field), 
          ('11i',id_field),('12i',id_field),('13i',id_field),('14i',id_field),('15i',id_field),
          ('01i',id_field),('02i',id_field),('03i',id_field),('04i',id_field),('05i',id_field)]

# TabularDataset
train, valid, test = TabularDataset.splits(path=datapath, train='train_a_new.csv', validation='valid_a_new.csv',
                                           test='test_a_new.csv', format='CSV', fields=fields, skip_header=True)

# Iterators
train_iter = BucketIterator(valid, batch_size=8, sort_key=lambda x: int(x.Id), shuffle=True,
                            device=device, train=True, sort=True, sort_within_batch=True)

valid_iter = BucketIterator(valid, batch_size=8, sort_key=lambda x: int(x.Id), shuffle=True,
                            device=device, train=True, sort=True, sort_within_batch=True)
test_iter = Iterator(test, batch_size=8, device=device, train=False, shuffle=False, sort=False)



#############
### model ###
#############

MODEL_NAME = 'bert-base-cased'

class BTIC(torch.nn.Module):
    def __init__(self):
        super(BTIC, self).__init__()
        self.tokenizer =  BertTokenizer.from_pretrained(MODEL_NAME)
        self.bertmodel_t = BertModel.from_pretrained(MODEL_NAME)
        self.transformer1= nn.Transformer(d_model= 768,nhead=8,
                                            num_encoder_layers = 6,
                                            num_decoder_layers =6,
                                            dim_feedforward = 516)
        self.transformer2= nn.Transformer(d_model= 768,nhead=8,
                                            num_encoder_layers = 6,
                                            num_decoder_layers =6,
                                            dim_feedforward = 516)
        self.attention = nn.MultiheadAttention(768, 8)
        self.fci1 = nn.Linear(2048, 1400)
        self.fci2 = nn.Linear(1400, 768)     
        self.pool1 = nn.MaxPool2d(kernel_size=(32, 32))
        self.pool2 = nn.AvgPool2d(kernel_size=(10, 1))
        self.fc = nn.Linear(24, 2)
        self.dropout = nn.Dropout(0.5)

    def text_read(self,Id):
        Idd = Id.cpu().numpy()
        tx = []
        for i in range(len(Idd)):
            j = Idd [i]
            ind = text[(text.Id==j)]
            ind = str(ind['body_text'].tolist())
            tx.append(ind)
        tx = tx
        tokens_pt = self.tokenizer(
                    tx,                      
                    add_special_tokens = True, 
                    max_length = 256,
                    truncation=True,
                    padding = 'max_length',
                    return_attention_mask = False, 
                    return_tensors = 'pt',     
                    ).to(device)
        outputs = self.bertmodel_t(**tokens_pt)
        last_hidden_state = outputs.last_hidden_state
        # pooler_output = outputs.pooler_output.unsqueeze(1)
        
        return  last_hidden_state    #pooler_output
    
    @staticmethod
    def img_read(Id):
        Idd = Id.cpu().numpy()
        imgg = []
        for i in range(len(Idd)):
            j = Idd [i]
            ind = image[(image.Id==j)]
            imgg.append(ind['img'].values.tolist())
        img = torch.tensor(imgg).to(device)
        img = img.squeeze(1)
        return img


    def textimage(self, Id):
        text = self.text_read(Id) 
        text = text.permute(1,0,2)
        img = self.img_read(Id)
        img = self.fci1(img)
        img = self.fci2(img)
        img = img.permute(1,0,2)
        img = self.transformer1(img,img)
        x = torch.cat((text,img),0)
        outputx, weights = self.attention(x,x,x)
        x = outputx.permute(1,0,2).unsqueeze(1)
        x = self.pool1(x)        
        x = self.pool2(x).squeeze(2).squeeze(1)
        return x

    def forward(self,Id):
        x = self.textimage(Id)
        x0 = self.dropout(x)
        x0 = self.fc(x0)               
        x0 = x0.squeeze(1)
        logit = torch.sigmoid(x0)
        
        return logit,x

###################
### memory bank ###
###################


def read_memory(Id, memory_bank):
    Idd = Id.cpu().numpy()
    memory = []
    for i in range(len(Idd)):
        j = Idd [i]
        n = memory_bank[memory_bank.Id==j].shape[0]
        if n == 0:
            memory_ = torch.zeros([1,24]).numpy().tolist()
            memory.append(memory_)
        else:
            memory_ = memory_bank[(memory_bank.Id==j)]
            memory__ = memory_['memory'].values.tolist()
            memory.append(memory__)  
    memory = torch.tensor(memory).to(device)
    memory = memory.squeeze(1)
    return memory

def multiread(id1,id2,id3,id4,id5,memory_bank):
    x1 = read_memory(id1,memory_bank)
    x2 = read_memory(id2,memory_bank)
    x3 = read_memory(id3,memory_bank)
    x4 = read_memory(id4,memory_bank)
    x5 = read_memory(id5,memory_bank)
    return x1,x2,x3,x4,x5
            
    

def write_memory(Id, x, memory_bank):
    Idd = Id.cpu().numpy()
    xx = x.detach().cpu().numpy()
    for i in range(len(Idd)):
        j = Idd [i]
        a = xx[i]   
        n = memory_bank[memory_bank.Id==j].shape[0]
        if n == 0:
            row = {'Id':j,'memory':a}
            memory_bank = memory_bank.append(row,ignore_index=True)
        else:
            b = memory_bank.loc[memory_bank['Id']==j].index.tolist()
            memory_bank = memory_bank.drop(b)
            row = {'Id':j,'memory':a}
            memory_bank = memory_bank.append(row,ignore_index=True)
    return memory_bank


#############
### train ###
#############


def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, train_loss_c_list, train_loss_s_list, valid_loss_list,valid_loss_c_list,valid_loss_s_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'train_loss_c_list':train_loss_c_list, 
                  'train_loss_s_list':train_loss_s_list,  
                  'valid_loss_list': valid_loss_list,
                  'valid_loss_c_list':valid_loss_c_list,
                  'valid_loss_s_list': valid_loss_s_list, 
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'],state_dict['train_loss_c_list'], state_dict['train_loss_s_list'], state_dict['valid_loss_list'], state_dict['valid_loss_c_list'],state_dict['valid_loss_s_list'], state_dict['global_steps_list']

def multisim(x,x1,x2,x3,x4,x5):
    s1 = (torch.cosine_similarity(x, x1, dim=1)+1)/2
    s2 = (torch.cosine_similarity(x, x2, dim=1)+1)/2
    s3 = (torch.cosine_similarity(x, x3, dim=1)+1)/2
    s4 = (torch.cosine_similarity(x, x4, dim=1)+1)/2
    s5 = (torch.cosine_similarity(x, x5, dim=1)+1)/2
    sim = (s1 + s2 + s3 +s4 +s5 )/5
    return sim

def consim(x,id1,id2,id3,id4,id5,memory_bank):
    x1,x2,x3,x4,x5 = multiread(id1, id2, id3, id4, id5, memory_bank)
    con_sim = multisim(x, x1, x2, x3, x4, x5) 
    return con_sim

# a is the weight alpha in the final loss 
def train(model,
          optimizer,
          criterion1 = nn.CrossEntropyLoss(),
          train_loader = train_iter,
          valid_loader = valid_iter,
          a = 0.2,
          num_epochs = 100,
          eval_every = len(train_iter) // 2,
          file_path = savepath,
          best_valid_loss = float("Inf")):
    
    running_loss = 0.0
    running_c_loss = 0.0
    running_s_loss =0.0
    valid_running_loss = 0.0
    valid_running_c_loss = 0.0
    valid_running_s_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    train_loss_c_list = []
    valid_loss_c_list = []
    train_loss_s_list = []
    valid_loss_s_list = []
    global_steps_list = []
    memory_bank= pd.DataFrame({'Id': [],'memory': []})



    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (Id,label,i11,i12,i13,i14,i15,i01,i02,i03,i04,i05), _ in train_loader:
            label = label.type(torch.LongTensor)           
            label = label.to(device)
            Id = Id.to(device)
            output, x= model(Id)
            memory_bank = write_memory(Id, x, memory_bank)
            sim1 = consim(x, i11, i12, i13, i14, i15, memory_bank)
            sim0 = consim(x, i01, i02, i03, i04, i05, memory_bank)
            sim = (sim0 - sim1)/2
            loss_c = criterion1(output, label)
            loss_s = ((label-1)*sim+ label*sim)+0.5
            loss_s = torch.mean(loss_s) 
            loss = (1-a)*loss_c + a*loss_s

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_c_loss += loss_c.item()
            running_s_loss += loss_s.item()
            global_step += 1

            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    

                    # validation loop
                    for (Id,label,i11,i12,i13,i14,i15,i01,i02,i03,i04,i05), _ in valid_loader:
                        label = label.type(torch.LongTensor)           
                        label = label.to(device)
                        Id = Id.to(device)
                        output, x= model(Id)
                        memory_bank = write_memory(Id, x, memory_bank)
                        sim1 = consim(x, i11, i12, i13, i14, i15, memory_bank)
                        sim0 = consim(x, i01, i02, i03, i04, i05, memory_bank)
                        sim = (sim0 - sim1)/2
                        loss_c = criterion1(output, label)
                        loss_s = ((label-1)*sim+ label*sim)+0.5
                        loss_s = torch.mean(loss_s) 
                        loss = (1-a)*loss_c + a*loss_s
                        valid_running_loss += loss.item()
                        valid_running_c_loss += loss_c.item()
                        valid_running_s_loss += loss_s.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                average_train_c_loss = running_c_loss / eval_every
                average_valid_c_loss = valid_running_c_loss / len(valid_loader)
                average_train_s_loss = running_s_loss / eval_every
                average_valid_s_loss = valid_running_s_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                train_loss_c_list.append(average_train_c_loss)
                valid_loss_c_list.append(average_valid_c_loss)
                train_loss_s_list.append(average_train_s_loss)
                valid_loss_s_list.append(average_valid_s_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                running_c_loss = 0.0                
                valid_running_c_loss = 0.0
                running_s_loss = 0.0                
                valid_running_s_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Train Loss (C): {:.4f},Train Loss (S): {:.4f},Valid Loss: {:.4f},Valid Loss (C)): {:.4f},Valid Loss (S): {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_train_c_loss, average_train_s_loss,
                              average_valid_loss, average_valid_c_loss, average_valid_s_loss))


                # checkpoint
                if best_valid_loss > average_valid_loss:
                   best_valid_loss = average_valid_loss
                   save_checkpoint(file_path + '/' + 'model_BTI&C.pt', model, best_valid_loss)
                   save_metrics(file_path + '/' + 'metrics_BTI&C.pt', train_loss_list, train_loss_c_list, train_loss_s_list, valid_loss_list, valid_loss_c_list,valid_loss_s_list, global_steps_list)
    
    save_metrics(file_path + '/' + 'metrics_BTI&C.pt', train_loss_list, train_loss_c_list, train_loss_s_list, valid_loss_list, valid_loss_c_list,valid_loss_s_list, global_steps_list)
    print('Finished Training!')


model = BTIC().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-6)

train(model=model, optimizer=optimizer)


train_loss_list, train_loss_c_list, train_loss_s_list, valid_loss_list, valid_loss_c_list, valid_loss_s_list,global_steps_list = load_metrics(savepath + '/metrics_BTI&C.pt')
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.plot(global_steps_list, train_loss_c_list, label='Train_criterion')
plt.plot(global_steps_list, valid_loss_c_list, label='Valid_criterion')
plt.plot(global_steps_list, train_loss_s_list, label='Train_similarity')
plt.plot(global_steps_list, valid_loss_s_list, label='Valid_similarity')
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.savefig(savepath+'/T&Vloss_BTI&C.jpg', dpi=300)
plt.show() 


#############
### test ###
#############

def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (Id,label,i11,i12,i13,i14,i15,i01,i02,i03,i04,i05), _ in test_loader:
            label = label.type(torch.LongTensor)           
            label = label.to(device)
            Id = Id.to(device)
            output, x = model(Id)
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(label.tolist())
    
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax= plt.subplot()
    p2 = sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['REAL', 'FAKE'])
    ax.yaxis.set_ticklabels(['REAL', 'FAKE'])
    s2 = p2.get_figure()
    s2.savefig(savepath+'/HeatMap_BTI&C.jpg',dpi=300)

best_model = BTIC().to(device)

load_checkpoint(savepath + '/model_BTI&C.pt', best_model)

evaluate(best_model, test_iter)
       


