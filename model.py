

import os
import math
import pickle

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from projection import *

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    longTensor = torch.cuda.LongTensor
    floatTensor = torch.cuda.FloatTensor

else:
    longTensor = torch.LongTensor
    floatTensor = torch.FloatTensor

class ANES(nn.Module):
    # def __init__(self,config):
    def __init__(self, config):
        super(ANES, self).__init__()

        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter

        self.embedding_size = config.embedding_size
        self.time_embedding_size = config.time_embedding_size
        self.cat_embedding_size=config.cat_embedding_size
        self.user_total = config.user_total
        self.POI_total = config.POI_total
        self.time_total = config.time_total
        self.cat_total=config.cat_total
        self.batch_size = config.batch_size
        user_time_weight = floatTensor(self.user_total, self.embedding_size)
        user_cat_weight=floatTensor(self.user_total,self.embedding_size)
        POI_time_weight=floatTensor(self.POI_total,self.embedding_size)
        POI_cat_weight=floatTensor(self.POI_total,self.embedding_size)
        nn.init.xavier_uniform_(user_time_weight)
        nn.init.xavier_uniform_(user_cat_weight)
        nn.init.xavier_uniform_(POI_cat_weight)
        nn.init.xavier_uniform_(POI_time_weight)

        time_tr_weight = floatTensor(self.time_total, self.time_embedding_size)
        time_proj_weight = floatTensor(self.time_total, self.time_embedding_size * self.embedding_size)
        cat_tr_weight = floatTensor(self.cat_total, self.cat_embedding_size)
        cat_proj_weight = floatTensor(self.cat_total, self.cat_embedding_size * self.embedding_size)

        # Use xavier initialization method to initialize embeddings of entities and relations
        nn.init.xavier_uniform_(time_tr_weight)
        nn.init.xavier_uniform_(time_proj_weight)
        nn.init.xavier_uniform_(cat_tr_weight)
        nn.init.xavier_uniform_(cat_proj_weight)

        self.user_time_embeddings = nn.Embedding(self.user_total, self.embedding_size)
        self.user_cat_embeddings = nn.Embedding(self.user_total, self.embedding_size)

        self.POI_time_embeddings = nn.Embedding(self.POI_total, self.embedding_size)
        self.POI_cat_embeddings = nn.Embedding(self.POI_total, self.embedding_size)


        self.time_tr_embeddings = nn.Embedding(self.time_total, self.time_embedding_size)
        self.time_proj_embeddings = nn.Embedding(self.time_total, self.time_embedding_size * self.embedding_size)
        self.cat_tr_embeddings = nn.Embedding(self.cat_total, self.time_embedding_size)
        self.cat_proj_embeddings = nn.Embedding(self.cat_total, self.time_embedding_size * self.embedding_size)

        self.user_time_embeddings.weight = nn.Parameter(user_time_weight)
        self.user_cat_embeddings.weight = nn.Parameter(user_cat_weight)
        self.POI_time_embeddings.weight = nn.Parameter(POI_time_weight)
        self.POI_cat_embeddings.weight = nn.Parameter(POI_cat_weight)

        self.time_tr_embeddings.weight = nn.Parameter(time_tr_weight)
        self.time_proj_embeddings.weight = nn.Parameter(time_proj_weight)
        self.cat_tr_embeddings.weight = nn.Parameter(cat_tr_weight)
        self.cat_proj_embeddings.weight = nn.Parameter(cat_proj_weight)


        normalize_user_time_emb = F.normalize(self.user_time_embeddings.weight.data, p=2, dim=1)
        normalize_user_cat_emb = F.normalize(self.user_cat_embeddings.weight.data, p=2, dim=1)
        normalize_POI_time_emb = F.normalize(self.POI_time_embeddings.weight.data, p=2, dim=1)
        normalize_POI_cat_emb = F.normalize(self.POI_cat_embeddings.weight.data, p=2, dim=1)

        normalize_time_tr_emb = F.normalize(self.time_tr_embeddings.weight.data, p=2, dim=1)
        normalize_time_proj_emb = F.normalize(self.time_proj_embeddings.weight.data, p=2, dim=1)
        normalize_cat_tr_emb = F.normalize(self.cat_tr_embeddings.weight.data, p=2, dim=1)
        normalize_cat_proj_emb = F.normalize(self.cat_proj_embeddings.weight.data, p=2, dim=1)


        self.user_time_embeddings.weight.data = normalize_user_time_emb
        self.user_cat_embeddings.weight.data = normalize_user_cat_emb


        self.POI_time_embeddings.weight.data = normalize_POI_time_emb
        self.POI_cat_embeddings.weight.data = normalize_POI_cat_emb


        self.time_tr_embeddings.weight.data = normalize_time_tr_emb
        self.time_proj_embeddings.weight.data = normalize_time_proj_emb
        self.cat_tr_embeddings.weight.data = normalize_cat_tr_emb
        self.cat_proj_embeddings.weight.data = normalize_cat_proj_emb
        self.LS=nn.LogSigmoid()

    def forward(self, pos_u, pos_t, pos_p, pos_c,neg_u, neg_t, neg_p, neg_u2,neg_c,neg_p2,NS):


        pos_u_time = self.user_time_embeddings(pos_u)
        pos_t_tr = self.time_tr_embeddings(pos_t)
        pos_t_proj = self.time_proj_embeddings(pos_t)
        pos_p_time = self.POI_time_embeddings(pos_p)
        pos_time_score = (projection_transR_pytorch(pos_u_time, pos_t_proj)+pos_t_tr)*pos_p_time

        pos_u_cat = self.user_cat_embeddings(pos_u)
        pos_c_tr = self.cat_tr_embeddings(pos_c)
        pos_c_proj = self.cat_proj_embeddings(pos_c)
        pos_p_cat = self.POI_cat_embeddings(pos_p)
        pos_cat_score = (projection_transR_pytorch(pos_u_cat, pos_c_proj)+pos_c_tr)*pos_p_cat

        #pos_p_e = projection_transR_pytorch(pos_p_e, pos_proj_spa)
        for i in range(len(neg_p)):  # i=[NS], repeat batchsize times
            nu = neg_u[i]
            nt = neg_t[i]
            np = neg_p[i]
            neg_u_time = self.user_time_embeddings(nu)
            neg_t_tr = self.time_tr_embeddings(nt)
            neg_t_proj = self.time_proj_embeddings(nt)
            neg_p_time = self.POI_time_embeddings(np)
            neg_time_score = (projection_transR_pytorch(neg_u_time, neg_t_proj) + neg_t_tr)*neg_p_time
            nu2 = neg_u2[i]
            np2 = neg_p2[i]
            nc = neg_c[i]
            neg_u_cat = self.user_cat_embeddings(nu2)
            neg_c_tr = self.cat_tr_embeddings(nc)
            neg_c_proj = self.cat_proj_embeddings(nc)
            neg_p_cat = self.POI_cat_embeddings(np2)
            neg_cat_score = (projection_transR_pytorch(neg_u_cat, neg_c_proj) + neg_c_tr)*neg_p_cat
            #neg_u_spa_e = projection_transR_pytorch(neg_u_spa, neg_proj_spa)
            #p_error = (neg_u_spa_e + neg_t_spa) * neg_p_e
            #p_error=-1*(neg_u_spa_e+neg_t_spa)*neg_p_e


            ## Edited : p_error ==> -1 * p_error, error /NS ==> error
            p_error = self.LS(torch.sum(-1*neg_time_score,1))+self.LS(torch.sum(-1*neg_cat_score,1))#((neg_u_spa_e + neg_t_spa)*neg_p_e))

            error = p_error

            #error = torch.sum(error)/NS
            error=torch.sum(error)
            try:
                neg = torch.cat([neg.view(-1), error.view(-1)])
            except:
                neg = error.view(-1)
        #pspa = pos_u_spa_e + pos_t_spa - pos_p_e
        #psem = pos_u_sem_e + pos_t_sem - pos_c_e

        pos_time_score = self.LS(torch.sum(pos_time_score, 1))
        pos_cat_score= self.LS(torch.sum(pos_cat_score, 1))

        #pos = alpha * torch.sum(pspa ** 2, dim=1) + (1 - alpha) * torch.sum(psem ** 2, dim=1)
        pos = pos_time_score+pos_cat_score
        pos=-1*pos
        neg=-1*neg
        return pos, neg
class ANES_GEO(nn.Module):
    # def __init__(self,config):
    def __init__(self, config):
        super(ANES_GEO, self).__init__()

        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter

        self.embedding_size = config.embedding_size
        self.time_embedding_size = config.time_embedding_size
        self.cat_embedding_size=config.cat_embedding_size
        self.geo_embedding_size=config.geo_embedding_size
        self.user_total = config.user_total
        self.POI_total = config.POI_total
        self.time_total = config.time_total
        self.geo_total = 100
        self.cat_total=config.cat_total
        self.batch_size = config.batch_size
        user_time_weight = floatTensor(self.user_total, self.embedding_size)
        user_cat_weight=floatTensor(self.user_total,self.embedding_size)
        user_geo_weight=floatTensor(self.user_total,self.embedding_size)

        POI_time_weight=floatTensor(self.POI_total,self.embedding_size)
        POI_cat_weight=floatTensor(self.POI_total,self.embedding_size)
        POI_geo_weight=floatTensor(self.POI_total,self.embedding_size)
        nn.init.xavier_uniform_(user_time_weight)
        nn.init.xavier_uniform_(user_cat_weight)
        nn.init.xavier_uniform_(user_geo_weight)
        nn.init.xavier_uniform_(POI_cat_weight)
        nn.init.xavier_uniform_(POI_time_weight)
        nn.init.xavier_uniform_(POI_geo_weight)

        time_tr_weight = floatTensor(self.time_total, self.time_embedding_size)
        time_proj_weight = floatTensor(self.time_total, self.time_embedding_size * self.embedding_size)
        cat_tr_weight = floatTensor(self.cat_total, self.cat_embedding_size)
        cat_proj_weight = floatTensor(self.cat_total, self.cat_embedding_size * self.embedding_size)
        geo_tr_weight = floatTensor(self.geo_total, self.geo_embedding_size)
        geo_proj_weight = floatTensor(self.geo_total, self.geo_embedding_size * self.embedding_size)

        # Use xavier initialization method to initialize embeddings of entities and relations
        nn.init.xavier_uniform_(time_tr_weight)
        nn.init.xavier_uniform_(time_proj_weight)
        nn.init.xavier_uniform_(cat_tr_weight)
        nn.init.xavier_uniform_(cat_proj_weight)
        nn.init.xavier_uniform_(geo_tr_weight)
        nn.init.xavier_uniform_(geo_proj_weight)

        self.user_time_embeddings = nn.Embedding(self.user_total, self.embedding_size)
        self.user_cat_embeddings = nn.Embedding(self.user_total, self.embedding_size)
        self.user_geo_embeddings = nn.Embedding(self.user_total, self.embedding_size)

        self.POI_time_embeddings = nn.Embedding(self.POI_total, self.embedding_size)
        self.POI_cat_embeddings = nn.Embedding(self.POI_total, self.embedding_size)
        self.POI_geo_embeddings = nn.Embedding(self.POI_total, self.embedding_size)


        self.time_tr_embeddings = nn.Embedding(self.time_total, self.time_embedding_size)
        self.time_proj_embeddings = nn.Embedding(self.time_total, self.time_embedding_size * self.embedding_size)
        self.cat_tr_embeddings = nn.Embedding(self.cat_total, self.cat_embedding_size)
        self.cat_proj_embeddings = nn.Embedding(self.cat_total, self.cat_embedding_size * self.embedding_size)
        self.geo_tr_embeddings = nn.Embedding(self.geo_total, self.geo_embedding_size)
        self.geo_proj_embeddings = nn.Embedding(self.geo_total, self.geo_embedding_size * self.embedding_size)

        self.user_time_embeddings.weight = nn.Parameter(user_time_weight)
        self.user_cat_embeddings.weight = nn.Parameter(user_cat_weight)
        self.user_geo_embeddings.weight = nn.Parameter(user_geo_weight)
        self.POI_time_embeddings.weight = nn.Parameter(POI_time_weight)
        self.POI_cat_embeddings.weight = nn.Parameter(POI_cat_weight)
        self.POI_geo_embeddings.weight = nn.Parameter(POI_geo_weight)

        self.time_tr_embeddings.weight = nn.Parameter(time_tr_weight)
        self.time_proj_embeddings.weight = nn.Parameter(time_proj_weight)
        self.cat_tr_embeddings.weight = nn.Parameter(cat_tr_weight)
        self.cat_proj_embeddings.weight = nn.Parameter(cat_proj_weight)
        self.geo_tr_embeddings.weight = nn.Parameter(geo_tr_weight)
        self.geo_proj_embeddings.weight = nn.Parameter(geo_proj_weight)

        normalize_user_time_emb = F.normalize(self.user_time_embeddings.weight.data, p=2, dim=1)
        normalize_user_cat_emb = F.normalize(self.user_cat_embeddings.weight.data, p=2, dim=1)
        normalize_user_geo_emb = F.normalize(self.user_geo_embeddings.weight.data, p=2, dim=1)
        normalize_POI_time_emb = F.normalize(self.POI_time_embeddings.weight.data, p=2, dim=1)
        normalize_POI_cat_emb = F.normalize(self.POI_cat_embeddings.weight.data, p=2, dim=1)
        normalize_POI_geo_emb = F.normalize(self.POI_cat_embeddings.weight.data, p=2, dim=1)

        normalize_time_tr_emb = F.normalize(self.time_tr_embeddings.weight.data, p=2, dim=1)
        normalize_time_proj_emb = F.normalize(self.time_proj_embeddings.weight.data, p=2, dim=1)
        normalize_cat_tr_emb = F.normalize(self.cat_tr_embeddings.weight.data, p=2, dim=1)
        normalize_cat_proj_emb = F.normalize(self.cat_proj_embeddings.weight.data, p=2, dim=1)
        normalize_geo_tr_emb = F.normalize(self.geo_tr_embeddings.weight.data, p=2, dim=1)
        normalize_geo_proj_emb = F.normalize(self.geo_proj_embeddings.weight.data, p=2, dim=1)


        self.user_time_embeddings.weight.data = normalize_user_time_emb
        self.user_cat_embeddings.weight.data = normalize_user_cat_emb
        self.user_geo_embeddings.weight.data = normalize_user_geo_emb


        self.POI_time_embeddings.weight.data = normalize_POI_time_emb
        self.POI_cat_embeddings.weight.data = normalize_POI_cat_emb
        self.POI_geo_embeddings.weight.data = normalize_POI_geo_emb


        self.time_tr_embeddings.weight.data = normalize_time_tr_emb
        self.time_proj_embeddings.weight.data = normalize_time_proj_emb
        self.cat_tr_embeddings.weight.data = normalize_cat_tr_emb
        self.cat_proj_embeddings.weight.data = normalize_cat_proj_emb
        self.geo_tr_embeddings.weight.data = normalize_geo_tr_emb
        self.geo_proj_embeddings.weight.data = normalize_geo_proj_emb
        self.LS=nn.LogSigmoid()

    def forward(self, pos_u, pos_t, pos_p, pos_c,pos_g,neg_u, neg_t, neg_p, neg_u2,neg_c,neg_p2,neg_u3,neg_g,neg_p3,NS):


        pos_u_time = self.user_time_embeddings(pos_u)
        pos_t_tr = self.time_tr_embeddings(pos_t)
        pos_t_proj = self.time_proj_embeddings(pos_t)
        pos_p_time = self.POI_time_embeddings(pos_p)
        pos_time_score = (projection_transR_pytorch(pos_u_time, pos_t_proj)+pos_t_tr)*pos_p_time

        pos_u_cat = self.user_cat_embeddings(pos_u)
        pos_c_tr = self.cat_tr_embeddings(pos_c)
        pos_c_proj = self.cat_proj_embeddings(pos_c)
        pos_p_cat = self.POI_cat_embeddings(pos_p)
        pos_cat_score = (projection_transR_pytorch(pos_u_cat, pos_c_proj)+pos_c_tr)*pos_p_cat

        pos_u_geo = self.user_geo_embeddings(pos_u)
        pos_g_tr = self.geo_tr_embeddings(pos_g)
        pos_g_proj = self.geo_proj_embeddings(pos_g)
        pos_p_geo = self.POI_geo_embeddings(pos_p)
        pos_geo_score = (projection_transR_pytorch(pos_u_geo, pos_g_proj)+pos_g_tr)*pos_p_geo

        #pos_p_e = projection_transR_pytorch(pos_p_e, pos_proj_spa)
        for i in range(len(neg_p)):  # i=[NS], repeat batchsize times
            nu = neg_u[i]
            nt = neg_t[i]
            np = neg_p[i]
            neg_u_time = self.user_time_embeddings(nu)
            neg_t_tr = self.time_tr_embeddings(nt)
            neg_t_proj = self.time_proj_embeddings(nt)
            neg_p_time = self.POI_time_embeddings(np)
            neg_time_score = (projection_transR_pytorch(neg_u_time, neg_t_proj) + neg_t_tr)*neg_p_time
            nu2 = neg_u2[i]
            np2 = neg_p2[i]
            nc = neg_c[i]
            neg_u_cat = self.user_cat_embeddings(nu2)
            neg_c_tr = self.cat_tr_embeddings(nc)
            neg_c_proj = self.cat_proj_embeddings(nc)
            neg_p_cat = self.POI_cat_embeddings(np2)
            neg_cat_score = (projection_transR_pytorch(neg_u_cat, neg_c_proj) + neg_c_tr)*neg_p_cat

            nu3=neg_u3[i]
            np3=neg_p3[i]
            ng=neg_g[i]
            neg_u_geo = self.user_geo_embeddings(nu3)
            neg_g_tr = self.geo_tr_embeddings(ng)
            neg_g_proj = self.geo_proj_embeddings(ng)
            neg_p_geo = self.POI_geo_embeddings(np3)
            neg_geo_score = (projection_transR_pytorch(neg_u_geo, neg_g_proj) + neg_g_tr)*neg_p_geo

            #neg_u_spa_e = projection_transR_pytorch(neg_u_spa, neg_proj_spa)
            #p_error = (neg_u_spa_e + neg_t_spa) * neg_p_e
            #p_error=-1*(neg_u_spa_e+neg_t_spa)*neg_p_e


            ## Edited : p_error ==> -1 * p_error, error /NS ==> error
            p_error = self.LS(torch.sum(-1*neg_time_score,1))+self.LS(torch.sum(-1*neg_cat_score,1))+self.LS(torch.sum(-1*neg_geo_score))#((neg_u_spa_e + neg_t_spa)*neg_p_e))

            error = p_error

            #error = torch.sum(error)/NS
            error=torch.sum(error)
            try:
                neg = torch.cat([neg.view(-1), error.view(-1)])
            except:
                neg = error.view(-1)
        #pspa = pos_u_spa_e + pos_t_spa - pos_p_e
        #psem = pos_u_sem_e + pos_t_sem - pos_c_e

        pos_time_score = self.LS(torch.sum(pos_time_score, 1))
        pos_cat_score= self.LS(torch.sum(pos_cat_score, 1))
        pos_geo_score= self.LS(torch.sum(pos_geo_score,1))
        #pos = alpha * torch.sum(pspa ** 2, dim=1) + (1 - alpha) * torch.sum(psem ** 2, dim=1)
        pos = pos_time_score+pos_cat_score+pos_geo_score
        pos=-1*pos
        neg=-1*neg
        return pos, neg
class ANES_GEO_YELP(nn.Module):
    # def __init__(self,config):
    def __init__(self, config):
        super(ANES_GEO_YELP, self).__init__()

        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter

        self.embedding_size = config.embedding_size
        self.cat_embedding_size=config.cat_embedding_size
        self.geo_embedding_size=config.geo_embedding_size
        self.user_total = config.user_total
        self.POI_total = config.POI_total
        self.geo_total = 100
        self.cat_total=config.cat_total
        self.batch_size = config.batch_size
        user_cat_weight=floatTensor(self.user_total,self.embedding_size)
        user_geo_weight=floatTensor(self.user_total,self.embedding_size)
        POI_cat_weight=floatTensor(self.POI_total,self.embedding_size)
        POI_geo_weight=floatTensor(self.POI_total,self.embedding_size)
        nn.init.xavier_uniform_(user_cat_weight)
        nn.init.xavier_uniform_(user_geo_weight)
        nn.init.xavier_uniform_(POI_cat_weight)
        nn.init.xavier_uniform_(POI_geo_weight)

        cat_tr_weight = floatTensor(self.cat_total, self.cat_embedding_size)
        cat_proj_weight = floatTensor(self.cat_total, self.cat_embedding_size * self.embedding_size)
        geo_tr_weight = floatTensor(self.geo_total, self.geo_embedding_size)
        geo_proj_weight = floatTensor(self.geo_total, self.geo_embedding_size * self.embedding_size)

        # Use xavier initialization method to initialize embeddings of entities and relations

        nn.init.xavier_uniform_(cat_tr_weight)
        nn.init.xavier_uniform_(cat_proj_weight)
        nn.init.xavier_uniform_(geo_tr_weight)
        nn.init.xavier_uniform_(geo_proj_weight)


        self.user_cat_embeddings = nn.Embedding(self.user_total, self.embedding_size)
        self.user_geo_embeddings = nn.Embedding(self.user_total, self.embedding_size)


        self.POI_cat_embeddings = nn.Embedding(self.POI_total, self.embedding_size)
        self.POI_geo_embeddings = nn.Embedding(self.POI_total, self.embedding_size)



        self.cat_tr_embeddings = nn.Embedding(self.cat_total, self.cat_embedding_size)
        self.cat_proj_embeddings = nn.Embedding(self.cat_total, self.cat_embedding_size * self.embedding_size)
        self.geo_tr_embeddings = nn.Embedding(self.geo_total, self.geo_embedding_size)
        self.geo_proj_embeddings = nn.Embedding(self.geo_total, self.geo_embedding_size * self.embedding_size)

        self.user_cat_embeddings.weight = nn.Parameter(user_cat_weight)
        self.user_geo_embeddings.weight = nn.Parameter(user_geo_weight)
        self.POI_cat_embeddings.weight = nn.Parameter(POI_cat_weight)
        self.POI_geo_embeddings.weight = nn.Parameter(POI_geo_weight)


        self.cat_tr_embeddings.weight = nn.Parameter(cat_tr_weight)
        self.cat_proj_embeddings.weight = nn.Parameter(cat_proj_weight)
        self.geo_tr_embeddings.weight = nn.Parameter(geo_tr_weight)
        self.geo_proj_embeddings.weight = nn.Parameter(geo_proj_weight)

        normalize_user_cat_emb = F.normalize(self.user_cat_embeddings.weight.data, p=2, dim=1)
        normalize_user_geo_emb = F.normalize(self.user_geo_embeddings.weight.data, p=2, dim=1)
        normalize_POI_cat_emb = F.normalize(self.POI_cat_embeddings.weight.data, p=2, dim=1)
        normalize_POI_geo_emb = F.normalize(self.POI_cat_embeddings.weight.data, p=2, dim=1)


        normalize_cat_tr_emb = F.normalize(self.cat_tr_embeddings.weight.data, p=2, dim=1)
        normalize_cat_proj_emb = F.normalize(self.cat_proj_embeddings.weight.data, p=2, dim=1)
        normalize_geo_tr_emb = F.normalize(self.geo_tr_embeddings.weight.data, p=2, dim=1)
        normalize_geo_proj_emb = F.normalize(self.geo_proj_embeddings.weight.data, p=2, dim=1)



        self.user_cat_embeddings.weight.data = normalize_user_cat_emb
        self.user_geo_embeddings.weight.data = normalize_user_geo_emb



        self.POI_cat_embeddings.weight.data = normalize_POI_cat_emb
        self.POI_geo_embeddings.weight.data = normalize_POI_geo_emb


        self.cat_tr_embeddings.weight.data = normalize_cat_tr_emb
        self.cat_proj_embeddings.weight.data = normalize_cat_proj_emb
        self.geo_tr_embeddings.weight.data = normalize_geo_tr_emb
        self.geo_proj_embeddings.weight.data = normalize_geo_proj_emb
        self.LS=nn.LogSigmoid()

    def forward(self, pos_u, pos_c,pos_p, pos_g, neg_u,neg_c,neg_p,neg_u2,neg_g,neg_p2,NS):

        #print('neg u ',neg_u)
        #print('neg c ',neg_c)
        #print('neg p ',neg_p)
        #print('neg u2 ',neg_u2)
        #print('neg g ',neg_g)
        #print('neg p2 ',neg_p2)

        pos_u_cat = self.user_cat_embeddings(pos_u)
        pos_c_tr = self.cat_tr_embeddings(pos_c)
        pos_c_proj = self.cat_proj_embeddings(pos_c)
        pos_p_cat = self.POI_cat_embeddings(pos_p)
        pos_cat_score = (projection_transR_pytorch(pos_u_cat, pos_c_proj)+pos_c_tr)*pos_p_cat

        pos_u_geo = self.user_geo_embeddings(pos_u)
        pos_g_tr = self.geo_tr_embeddings(pos_g)
        pos_g_proj = self.geo_proj_embeddings(pos_g)
        pos_p_geo = self.POI_geo_embeddings(pos_p)
        pos_geo_score = (projection_transR_pytorch(pos_u_geo, pos_g_proj)+pos_g_tr)*pos_p_geo

        #pos_p_e = projection_transR_pytorch(pos_p_e, pos_proj_spa)
        for i in range(len(neg_p)):  # i=[NS], repeat batchsize times
            nu = neg_u[i]
            np = neg_p[i]
            nc = neg_c[i]
            neg_u_cat = self.user_cat_embeddings(nu)
            neg_c_tr = self.cat_tr_embeddings(nc)
            neg_c_proj = self.cat_proj_embeddings(nc)
            neg_p_cat = self.POI_cat_embeddings(np)
            neg_cat_score = (projection_transR_pytorch(neg_u_cat, neg_c_proj) + neg_c_tr)*neg_p_cat

            nu2=neg_u2[i]
            np2=neg_p2[i]
            ng=neg_g[i]
            neg_u_geo = self.user_geo_embeddings(nu2)
            neg_g_tr = self.geo_tr_embeddings(ng)
            neg_g_proj = self.geo_proj_embeddings(ng)
            neg_p_geo = self.POI_geo_embeddings(np2)
            #print('nu',nu)
            #print('nc',nc)
            #print('np',np)
            #print('nu2',nu2)
            #print('ng',ng)
            #print('np2',np2)
            neg_geo_score = (projection_transR_pytorch(neg_u_geo, neg_g_proj) + neg_g_tr)*neg_p_geo

            #neg_u_spa_e = projection_transR_pytorch(neg_u_spa, neg_proj_spa)
            #p_error = (neg_u_spa_e + neg_t_spa) * neg_p_e
            #p_error=-1*(neg_u_spa_e+neg_t_spa)*neg_p_e


            ## Edited : p_error ==> -1 * p_error, error /NS ==> error
            p_error = self.LS(torch.sum(-1*neg_cat_score,1))+self.LS(torch.sum(-1*neg_geo_score))#((neg_u_spa_e + neg_t_spa)*neg_p_e))

            error = p_error

            #error = torch.sum(error)/NS
            error=torch.sum(error)
            try:
                neg = torch.cat([neg.view(-1), error.view(-1)])
            except:
                neg = error.view(-1)
        #pspa = pos_u_spa_e + pos_t_spa - pos_p_e
        #psem = pos_u_sem_e + pos_t_sem - pos_c_e

        pos_cat_score= self.LS(torch.sum(pos_cat_score, 1))
        pos_geo_score= self.LS(torch.sum(pos_geo_score,1))
        #pos = alpha * torch.sum(pspa ** 2, dim=1) + (1 - alpha) * torch.sum(psem ** 2, dim=1)
        pos = pos_cat_score+pos_geo_score
        pos=-1*pos
        neg=-1*neg
        return pos, neg
class ANES_GEO_Others(nn.Module):
    # def __init__(self,config):
    def __init__(self, config):
        super(ANES_GEO_Others, self).__init__()

        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter

        self.embedding_size = config.embedding_size
        self.time_embedding_size = config.time_embedding_size
        self.geo_embedding_size=config.geo_embedding_size
        self.user_total = config.user_total
        self.POI_total = config.POI_total
        self.time_total = config.time_total
        self.geo_total = 100
        self.batch_size = config.batch_size
        user_time_weight = floatTensor(self.user_total, self.embedding_size)
        user_geo_weight=floatTensor(self.user_total,self.embedding_size)

        POI_time_weight=floatTensor(self.POI_total,self.embedding_size)
        POI_geo_weight=floatTensor(self.POI_total,self.embedding_size)
        nn.init.xavier_uniform_(user_time_weight)
        nn.init.xavier_uniform_(user_geo_weight)
        nn.init.xavier_uniform_(POI_time_weight)
        nn.init.xavier_uniform_(POI_geo_weight)

        time_tr_weight = floatTensor(self.time_total, self.time_embedding_size)
        time_proj_weight = floatTensor(self.time_total, self.time_embedding_size * self.embedding_size)
        geo_tr_weight = floatTensor(self.geo_total, self.geo_embedding_size)
        geo_proj_weight = floatTensor(self.geo_total, self.geo_embedding_size * self.embedding_size)

        # Use xavier initialization method to initialize embeddings of entities and relations
        nn.init.xavier_uniform_(time_tr_weight)
        nn.init.xavier_uniform_(time_proj_weight)
        nn.init.xavier_uniform_(geo_tr_weight)
        nn.init.xavier_uniform_(geo_proj_weight)

        self.user_time_embeddings = nn.Embedding(self.user_total, self.embedding_size)
        self.user_geo_embeddings = nn.Embedding(self.user_total, self.embedding_size)

        self.POI_time_embeddings = nn.Embedding(self.POI_total, self.embedding_size)
        self.POI_geo_embeddings = nn.Embedding(self.POI_total, self.embedding_size)


        self.time_tr_embeddings = nn.Embedding(self.time_total, self.time_embedding_size)
        self.time_proj_embeddings = nn.Embedding(self.time_total, self.time_embedding_size * self.embedding_size)
        self.geo_tr_embeddings = nn.Embedding(self.geo_total, self.geo_embedding_size)
        self.geo_proj_embeddings = nn.Embedding(self.geo_total, self.geo_embedding_size * self.embedding_size)

        self.user_time_embeddings.weight = nn.Parameter(user_time_weight)
        self.user_geo_embeddings.weight = nn.Parameter(user_geo_weight)
        self.POI_time_embeddings.weight = nn.Parameter(POI_time_weight)
        self.POI_geo_embeddings.weight = nn.Parameter(POI_geo_weight)

        self.time_tr_embeddings.weight = nn.Parameter(time_tr_weight)
        self.time_proj_embeddings.weight = nn.Parameter(time_proj_weight)
        self.geo_tr_embeddings.weight = nn.Parameter(geo_tr_weight)
        self.geo_proj_embeddings.weight = nn.Parameter(geo_proj_weight)

        normalize_user_time_emb = F.normalize(self.user_time_embeddings.weight.data, p=2, dim=1)
        normalize_user_geo_emb = F.normalize(self.user_geo_embeddings.weight.data, p=2, dim=1)
        normalize_POI_time_emb = F.normalize(self.POI_time_embeddings.weight.data, p=2, dim=1)
        normalize_POI_geo_emb = F.normalize(self.POI_geo_embeddings.weight.data, p=2, dim=1)

        normalize_time_tr_emb = F.normalize(self.time_tr_embeddings.weight.data, p=2, dim=1)
        normalize_time_proj_emb = F.normalize(self.time_proj_embeddings.weight.data, p=2, dim=1)
        normalize_geo_tr_emb = F.normalize(self.geo_tr_embeddings.weight.data, p=2, dim=1)
        normalize_geo_proj_emb = F.normalize(self.geo_proj_embeddings.weight.data, p=2, dim=1)


        self.user_time_embeddings.weight.data = normalize_user_time_emb
        self.user_geo_embeddings.weight.data = normalize_user_geo_emb


        self.POI_time_embeddings.weight.data = normalize_POI_time_emb
        self.POI_geo_embeddings.weight.data = normalize_POI_geo_emb


        self.time_tr_embeddings.weight.data = normalize_time_tr_emb
        self.time_proj_embeddings.weight.data = normalize_time_proj_emb
        self.geo_tr_embeddings.weight.data = normalize_geo_tr_emb
        self.geo_proj_embeddings.weight.data = normalize_geo_proj_emb
        self.LS=nn.LogSigmoid()

    def forward(self, pos_u, pos_t, pos_p, pos_g,neg_u, neg_t, neg_p, neg_u2,neg_g,neg_p2,NS):


        pos_u_time = self.user_time_embeddings(pos_u)
        pos_t_tr = self.time_tr_embeddings(pos_t)
        pos_t_proj = self.time_proj_embeddings(pos_t)
        pos_p_time = self.POI_time_embeddings(pos_p)
        pos_time_score = (projection_transR_pytorch(pos_u_time, pos_t_proj)+pos_t_tr)*pos_p_time

        pos_u_geo = self.user_geo_embeddings(pos_u)
        pos_g_tr = self.geo_tr_embeddings(pos_g)
        pos_g_proj = self.geo_proj_embeddings(pos_g)
        pos_p_geo = self.POI_geo_embeddings(pos_p)
        pos_geo_score = (projection_transR_pytorch(pos_u_geo, pos_g_proj)+pos_g_tr)*pos_p_geo

        #pos_p_e = projection_transR_pytorch(pos_p_e, pos_proj_spa)
        for i in range(len(neg_p)):  # i=[NS], repeat batchsize times
            nu = neg_u[i]
            nt = neg_t[i]
            np = neg_p[i]
            neg_u_time = self.user_time_embeddings(nu)
            neg_t_tr = self.time_tr_embeddings(nt)
            neg_t_proj = self.time_proj_embeddings(nt)
            neg_p_time = self.POI_time_embeddings(np)
            neg_time_score = (projection_transR_pytorch(neg_u_time, neg_t_proj) + neg_t_tr)*neg_p_time
            nu2=neg_u2[i]
            np2=neg_p2[i]
            ng=neg_g[i]
            neg_u_geo = self.user_geo_embeddings(nu2)
            neg_g_tr = self.geo_tr_embeddings(ng)
            neg_g_proj = self.geo_proj_embeddings(ng)
            neg_p_geo = self.POI_geo_embeddings(np2)
            neg_geo_score = (projection_transR_pytorch(neg_u_geo, neg_g_proj) + neg_g_tr)*neg_p_geo

            #neg_u_spa_e = projection_transR_pytorch(neg_u_spa, neg_proj_spa)
            #p_error = (neg_u_spa_e + neg_t_spa) * neg_p_e
            #p_error=-1*(neg_u_spa_e+neg_t_spa)*neg_p_e


            ## Edited : p_error ==> -1 * p_error, error /NS ==> error
            p_error = self.LS(torch.sum(-1*neg_time_score,1))+self.LS(torch.sum(-1*neg_geo_score))#((neg_u_spa_e + neg_t_spa)*neg_p_e))

            error = p_error

            #error = torch.sum(error)/NS
            error=torch.sum(error)
            try:
                neg = torch.cat([neg.view(-1), error.view(-1)])
            except:
                neg = error.view(-1)
        #pspa = pos_u_spa_e + pos_t_spa - pos_p_e
        #psem = pos_u_sem_e + pos_t_sem - pos_c_e

        pos_time_score = self.LS(torch.sum(pos_time_score, 1))
        pos_geo_score= self.LS(torch.sum(pos_geo_score,1))
        #pos = alpha * torch.sum(pspa ** 2, dim=1) + (1 - alpha) * torch.sum(psem ** 2, dim=1)
        pos = pos_time_score+pos_geo_score
        pos=-1*pos
        neg=-1*neg
        return pos, neg
class LBGC_v4(nn.Module):
    # def __init__(self,config):
    def __init__(self, config, latlon, UE_spa, PE, pretrained):
        super(LBGC_v4, self).__init__()
        if pretrained==1:
            print("Use Pretrained Result")
        else:
            print("Use Random Embedding")

        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter

        self.embedding_size = config.embedding_size
        self.time_embedding_size = config.time_embedding_size

        self.user_total = config.user_total
        self.POI_total = config.POI_total
        self.time_total = config.time_total

        self.batch_size = config.batch_size
        if pretrained==1:
            POI_weight = PE
            user_spatial_weight = UE_spa
        else:
            user_spatial_weight = floatTensor(self.user_total, self.embedding_size)
            POI_weight=floatTensor(self.POI_total,self.embedding_size)
            nn.init.xavier_uniform_(user_spatial_weight)
            nn.init.xavier_uniform_(POI_weight)

        time_spatial_weight = floatTensor(self.time_total, self.time_embedding_size)
        proj_spatial_weight = floatTensor(self.time_total, self.time_embedding_size * self.embedding_size)

        # Use xavier initialization method to initialize embeddings of entities and relations
        nn.init.xavier_uniform_(time_spatial_weight)
        nn.init.xavier_uniform_(proj_spatial_weight)

        self.user_spatial_embeddings = nn.Embedding(self.user_total, self.embedding_size)
        self.POI_embeddings = nn.Embedding(self.POI_total, self.embedding_size)

        self.time_spatial_embeddings = nn.Embedding(self.time_total, self.time_embedding_size)

        self.proj_spatial_embeddings = nn.Embedding(self.time_total, self.time_embedding_size * self.embedding_size)

        self.user_spatial_embeddings.weight = nn.Parameter(user_spatial_weight)
        self.POI_embeddings.weight = nn.Parameter(POI_weight)
        self.time_spatial_embeddings.weight = nn.Parameter(time_spatial_weight)
        self.proj_spatial_embeddings.weight = nn.Parameter(proj_spatial_weight)

        normalize_user_spatial_emb = F.normalize(self.user_spatial_embeddings.weight.data, p=2, dim=1)
        normalize_POI_emb = F.normalize(self.POI_embeddings.weight.data, p=2, dim=1)
        normalize_time_spatial_emb = F.normalize(self.time_spatial_embeddings.weight.data, p=2, dim=1)
        normalize_proj_spatial_emb = F.normalize(self.proj_spatial_embeddings.weight.data, p=2, dim=1)

        self.user_spatial_embeddings.weight.data = normalize_user_spatial_emb
        self.POI_embeddings.weight.data = normalize_POI_emb
        self.time_spatial_embeddings.weight.data = normalize_time_spatial_emb
        self.proj_spatial_embeddings.weight.data = normalize_proj_spatial_emb
        self.LS=nn.LogSigmoid()

    def forward(self, pos_u, pos_t, pos_p, neg_u, neg_t, neg_p, NS):
        pos_u_spa = self.user_spatial_embeddings(pos_u)

        pos_t_spa = self.time_spatial_embeddings(pos_t)

        pos_p_e = self.POI_embeddings(pos_p)

        pos_proj_spa = self.proj_spatial_embeddings(pos_t)

        pos_u_spa_e = projection_transR_pytorch(pos_u_spa, pos_proj_spa)
        #pos_p_e = projection_transR_pytorch(pos_p_e, pos_proj_spa)
        for i in range(len(neg_p)):  # i=[NS], repeat batchsize times
            nu = neg_u[i]
            nt = neg_t[i]
            np = neg_p[i]
            neg_u_spa = self.user_spatial_embeddings(nu)
            neg_t_spa = self.time_spatial_embeddings(nt)
            neg_p_e = self.POI_embeddings(np)

            neg_proj_spa = self.proj_spatial_embeddings(nt)
            neg_u_spa_e = projection_transR_pytorch(neg_u_spa, neg_proj_spa)
            #p_error = (neg_u_spa_e + neg_t_spa) * neg_p_e
            p_error=-1*(neg_u_spa_e+neg_t_spa)*neg_p_e
            ## Edited : p_error ==> -1 * p_error, error /NS ==> error
            p_error = self.LS(torch.sum(p_error,1))#((neg_u_spa_e + neg_t_spa)*neg_p_e))

            error = p_error

            #error = torch.sum(error)/NS
            error=torch.sum(error)
            try:
                neg = torch.cat([neg.view(-1), error.view(-1)])
            except:
                neg = error.view(-1)
        #pspa = pos_u_spa_e + pos_t_spa - pos_p_e
        #psem = pos_u_sem_e + pos_t_sem - pos_c_e
        pspa = (pos_u_spa_e + pos_t_spa) * pos_p_e
        pspa=self.LS(torch.sum(pspa,1))

        #pos = alpha * torch.sum(pspa ** 2, dim=1) + (1 - alpha) * torch.sum(psem ** 2, dim=1)
        pos = pspa
        pos=-1*pos
        neg=-1*neg
        return pos, neg


