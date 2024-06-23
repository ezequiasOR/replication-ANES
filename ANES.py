import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn
import easydict
import argparse

import numpy as np
import time
import datetime
import random
from utils import LBSN
from utils import *
from data import *
# from evaluation import *
import loss
import model
import pandas as pd
import matplotlib.pyplot as plt
import collections
import math
from sklearn.manifold import TSNE

###    Hyperarameters
MARGIN  = 3
FRIENDSHIP = 1
N=5
TRAIN_TIMES=5000
stime=time.time()
RATIO=0.2
def ctime():
    return round(time.time()-stime,5)
Location='NYC_100'
dataset=""
print("ANES Using ",Location,"Data From",dataset,)
args = easydict.EasyDict({
    "dataset":'./'+Location,
    "learning_rate":0.0001,
    "early_stopping_round":0,
    "L1_flag":True,
        'embedding_size': 128,
        'time_embedding_size': 128,
        'cat_embedding_size':128,
        'geo_embedding_size':128,
        'margin': MARGIN,
        'filter': 1,
        'momentum':0.5,
        'seed': 0,
        'optimizer': 1,
        'loss_type': 0,
        'num_batches':1447,
        'train_times':TRAIN_TIMES,
        })

model_name="GEO_"+Location


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU


def allround(x):
    y = []
    for i in x:
        y.append(round(i, 3))
    return y


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    longTensor = torch.cuda.LongTensor
    floatTensor = torch.cuda.FloatTensor
else:
    longTensor = torch.LongTensor
    floatTensor = torch.FloatTensor

def dist(ST, L):
    D = 0
    EucD = 0
    SST = 0
    SL = 0
    for i in range(len(ST)):
        SST += ST[i] * ST[i]
        SL += L[i] * L[i]
        D += ST[i] * L[i]
        EucD += (ST[i] - L[i]) * (ST[i] - L[i])
    euc=D
    try:
        cos=D/math.sqrt(SL*SST)
    except:
        cos=1

    return np.array([D,cos,EucD])
    # return round(math.sqrt(EucD),3)
    # return cos(ST,L)
    # return round(math.sqrt(EucD),3)

class Config(object):
    # INIT value 값 안 바뀜(초기화임)
    def __init__(self):
        self.dataset = None
        self.learning_rate = 0.001
        self.early_stopping_round = 0
        self.L1_flag = True
        self.embedding_size = 100
        self.time_embedding_size = 50
        self.geo_embedding_size=50
        self.num_batches = 100
        self.train_times = 1000
        self.margin = 5
        self.filter = True
        self.momentum = 0.9
        self.optimizer = optim.Adam
        self.loss_function = loss.marginLoss
        self.user_total = 0
        self.POI_total = 0
        self.time_total = 0
        self.cat_total=0
        self.category_total = 0
        self.batch_size = 0

"""

Friendship Get, 

"""
ff = open(args.dataset + "/user2id.txt", 'r')
conv_ID = dict()
for i in ff.readlines()[1:]:
    a, b = i.strip().split('\t')
    conv_ID[a] = b
fr_old = dict()
fr_new = dict()
# File 안에 있는건 로우 아이디니까 conv_ID로 변환해서 프렌드쉽 바꿔줌
users = np.arange(0, len(conv_ID))
valid_friendship=[]
test_friendship=[]
print("Friendship Processing...",ctime())
f=open(args.dataset+"/false_pairs_val.txt",'r')
f2=open(args.dataset+"/true_pairs_val.txt",'r')
for i in f.readlines():
    a,b=i.split('\t')
    valid_friendship.append((a,b,0))
for i in f2.readlines():
    a,b=i.split('\t')
    valid_friendship.append((a,b,1))
f3=open(args.dataset+"/false_pairs_test.txt",'r')
f4=open(args.dataset+"/true_pairs_test.txt",'r')
for i in f3.readlines():
    a,b=i.split('\t')
    test_friendship.append((a,b,0))
for i in f4.readlines():
    a,b=i.split('\t')
    test_friendship.append((a,b,1))

print("Friendship Processing Done", ctime())
def F_validation(instance, model):
    # Dataset에 대한 RoCAUC 출력
    u_sp = model.user_time_embeddings.weight.data
    u_cat = model.user_cat_embeddings.weight.data
    u_geo = model.user_geo_embeddings.weight.data
    Y_score_dot = []
    Y_score_cos=[]
    Y_score_euc=[]
    Y_true = []
    for (a, b, flag) in instance:
        a = int(a)
        b = int(b)
        flag = int(flag)
        X1 = u_sp[a].tolist()
        Y1 = u_sp[b].tolist()
        X2 = u_cat[a].tolist()
        Y2 = u_cat[b].tolist()
        X3 = u_geo[a].tolist()
        Y3 = u_geo[b].tolist()
        [dot,cos,euc] = dist(X1, Y1)
        [dot2,cos2,euc2] = dist(X2, Y2)
        [dot3,cos3,euc3] = dist(X3,Y3)
        Y_score_dot.append(dot+dot2+dot3)
        Y_score_cos.append(cos+cos2+cos3)
        Y_score_euc.append(euc+euc2+euc3)
        Y_true.append(flag)
    dot_score=sklearn.metrics.roc_auc_score(Y_true,Y_score_dot)
    cos_score=sklearn.metrics.roc_auc_score(Y_true,Y_score_cos)
    euc_score=1-sklearn.metrics.roc_auc_score(Y_true,Y_score_euc)

    return np.array([dot_score,cos_score,euc_score])

argparser = argparse.ArgumentParser()

trainTotal, trainList, trainDict = loadLBSN(args.dataset, 'alldata2id_geo.txt')
LBSNTotal, LBSNList, LBSNDict = loadLBSN(args.dataset, 'alldata2id_geo.txt')
utpDict={}
ucpDict={}
ugpDict={}
POIs={}
POIs[0]=0
Pdist=[0]
CattoPOI={}
POItoCat={}
POItoGrid={}
Users={}
Udist=[0]

Cdist=[0]
Categories={}
for lbsn in LBSNList:
    [u,t,p,c,g]=[lbsn.u,lbsn.t,lbsn.p,lbsn.c,lbsn.g]
    utpDict[(u,t,p)]=True
    ucpDict[(u,c,p)]=True
    ugpDict[(u,g,p)]=True
    if p in POIs: POIs[p]+=1
    else: POIs[p]=1
    if u in Users: Users[u]+=1
    else: Users[u]=1
    if c not in CattoPOI:
        CattoPOI[c]=[]
    if p not in CattoPOI[c]:
        CattoPOI[c].append(p)
    if p not in POItoCat:
        POItoCat[p]=c
    if p not in POItoGrid:
        POItoGrid[p]=g


#LBSNTotal 로 나누면 됨
POIs[0]=0
for i in range(len(POIs)):
    Pdist.append(math.pow(POIs[i],0.75))
    if i>0: Pdist[-1]=Pdist[-2]+Pdist[-1]
    #if i>0: Pdist[-1]=Pdist[-2]+Pdist[-1]
for i in range(len(Users)):
    Udist.append(math.pow(Users[i],0.75))
    if i>0: Udist[-1]=Udist[-2]+Udist[-1]

sumP=Pdist[-1]
sumU=Udist[-1]
for i in range(len(POIs)+1):
    Pdist[i]=Pdist[i]/sumP
for i in range(len(Users)+1):
    Udist[i]=Udist[i]/sumU
#headstart,headend,tailstart,tailend= divide_head_and_tail('./data/' + args.dataset, 'entity2id.txt')
config = Config()
config.dataset = args.dataset
config.learning_rate = args.learning_rate

config.early_stopping_round = args.early_stopping_round

if args.L1_flag == 1:
    config.L1_flag = True
else:
    config.L1_flag = False

config.embedding_size = args.embedding_size
config.time_embedding_size = args.time_embedding_size
config.cat_embedding_size=args.cat_embedding_size
config.geo_embedding_size=args.geo_embedding_size
config.num_batches = args.num_batches
config.train_times = args.train_times
config.margin = args.margin

if args.filter == 1:
    config.filter = True
else:
    config.filter = False

config.momentum = args.momentum

if args.optimizer == 0:
    config.optimizer = optim.SGD
elif args.optimizer == 1:
    config.optimizer = optim.Adam
elif args.optimizer == 2:
    config.optimizer = optim.RMSprop

if args.loss_type == 0:
    config.loss_function = loss.marginLoss

config.user_total = getAnythingTotal(config.dataset, 'user2id.txt')
config.POI_total=getAnythingTotal( config.dataset, 'POI2id.txt')
config.time_total = 168
config.cat_total=int(open(args.dataset+'/category2id.txt','r').readlines()[0].strip())
config.batch_size = trainTotal // config.num_batches

# f_latlon=open(args.dataset+"/latlon.txt",'r').readlines()
try:
    X = pd.read_csv(args.dataset + '/latlon.txt', delimiter='\t', header=None)
    latlon = floatTensor(X.values).to(device)
except:
    latlon=0

loss_function = config.loss_function()

print("Model Loading...",ctime())
model = model.ANES_GEO(config)
iter_zero=-1
#modelpath='./models/POINYC_100_128_62_3.pt'
#iter_zero = int(modelpath.split('_')[3])
#print("Resuming Training, Start from epoch",iter_zero)
#model.load_state_dict(torch.load(modelpath))
Trace=[]
if USE_CUDA:
    model.cuda()
    loss_function.cuda()

optimizer = config.optimizer(model.parameters(), lr=config.learning_rate,weight_decay=0.001)
margin = autograd.Variable(floatTensor([config.margin]))


print("Model loaded, learning start...",ctime())
trainBatchList = getBatchList(trainList, config.num_batches)

best_loss=100
best_epoch=0

X=[]
Y=[]
Y2=[]
best=0
for epoch in range(iter_zero+1,config.train_times):
#for epoch in range(1):
    total_loss = floatTensor([0.0])
    random.shuffle(trainBatchList)
    for batchList in trainBatchList:
        if config.filter == True:
            pos_u_batch,pos_t_batch,pos_p_batch,pos_c_batch,pos_g_batch,neg_u_batch,neg_t_batch,neg_p_batch,\
                neg_u_batch2,neg_c_batch,neg_p_batch2,neg_u_batch3,neg_g_batch,neg_p_batch3=getLBSNBatch_Foursquare_GEO(batchList, config.batch_size, utpDict,ucpDict,ugpDict,Udist,Pdist,POItoCat,POItoGrid,N,RATIO)
            #pos_u_batch, pos_t_batch, pos_p_batch, pos_c_batch,neg_u_batch,neg_t_batch,neg_p_batch,neg_c_batch = \
            #getLBSNBatch_v2(batchList, config.batch_size, utpDict,ucpDict,Udist,Pdist,CattoPOI,N,RATIO)

        pos_u_batch = autograd.Variable(longTensor(pos_u_batch))
        pos_t_batch = autograd.Variable(longTensor(pos_t_batch))
        pos_p_batch = autograd.Variable(longTensor(pos_p_batch))
        pos_c_batch = autograd.Variable(longTensor(pos_c_batch))
        pos_g_batch = autograd.Variable(longTensor(pos_g_batch))

        for i in range(len(neg_u_batch)):
            neg_u_batch[i] = autograd.Variable(longTensor(neg_u_batch[i]))
            neg_t_batch[i] = autograd.Variable(longTensor(neg_t_batch[i]))
            neg_p_batch[i] = autograd.Variable(longTensor(neg_p_batch[i]))
            neg_u_batch2[i] = autograd.Variable(longTensor(neg_u_batch2[i]))
            neg_c_batch[i] = autograd.Variable(longTensor(neg_c_batch[i]))
            neg_p_batch2[i] = autograd.Variable(longTensor(neg_p_batch2[i]))
            neg_u_batch3[i] = autograd.Variable(longTensor(neg_u_batch3[i]))
            neg_g_batch[i] = autograd.Variable(longTensor(neg_g_batch[i]))
            neg_p_batch3[i] = autograd.Variable(longTensor(neg_p_batch3[i]))

        model.zero_grad()
        pos, neg= model(pos_u_batch,
            pos_t_batch, pos_p_batch, pos_c_batch,pos_g_batch,neg_u_batch, neg_t_batch,neg_p_batch, neg_u_batch2,neg_c_batch,neg_p_batch2,neg_u_batch3,neg_g_batch,neg_p_batch3,N)
        if args.loss_type == 0:
            losses = loss_function(pos, neg, margin)
        else:
            losses = loss_function(pos, neg)
        #spa_embeddings = model.user_spatial_embeddings(torch.cat([pos_u_batch,torch.cat(neg_u_batch)]))
        #poi_embeddings = model.POI_embeddings(torch.cat([pos_p_batch,torch.cat(neg_p_batch)]))
        #proj_embeddings = model.proj_spatial_embeddings(torch.cat([pos_t_batch,torch.cat(neg_t_batch)]))
        #time_embeddings = model.time_spatial_embeddings(torch.cat([pos_t_batch,torch.cat(neg_t_batch)]))
        #reg=loss.normLoss(spa_embeddings)+loss.normLoss(poi_embeddings)+ \
        #loss.normLoss(proj_embeddings)+loss.normLoss(time_embeddings)
        #losses=losses+0.01*reg
        losses.backward()
        optimizer.step()
        total_loss += losses.data

    if epoch%10==0:
        print("Epoch ",epoch)
    if epoch%1==0:
        print("Epoch",epoch,'Loss:',round(total_loss.item()/len(trainList),6))
        f_result_val=[round(v,3) for v in F_validation(valid_friendship,model)]
        f_result_test=[round(v,3) for v in F_validation(test_friendship,model)]
        print(f_result_val,f_result_test, '<',ctime(),'>')
        PATH='./models/'+model_name+'_'+str(args.embedding_size)+'_'+str(epoch)+'_'+str(MARGIN)+'.pt'
        print('Save to ', PATH)
        if max(f_result_test)>=best:
            print("Best, Model Saving...",ctime())
            torch.save(model.state_dict(),PATH)
            print("Model save done",ctime())
            best=max(f_result_test)
            for e in range(epoch-3,epoch):
                PATH = './models/' + model_name + '_' + str(args.embedding_size) + '_' + str(e) + '_' + str(MARGIN) + '.pt'
                try:
                    os.remove(PATH)
                except:
                    continue
            print("Removing old model done",ctime())
