

import os
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor

def projection_transH(original, norm):
	# numpy version
	return original - np.sum(original * norm, axis=1, keepdims=True) * norm

def projection_transH_pytorch(original, norm):
	return original - torch.sum(original * norm, dim=1, keepdim=True) * norm

def projection_transR_pytorch(original, proj_matrix):
	#print('Before:',original.shape,proj_matrix.shape) # (148,100) , (148,5000)
	ent_embedding_size = original.shape[1] # 256
	rel_embedding_size = proj_matrix.shape[1] // ent_embedding_size # 64
	original = original.view(-1,1,ent_embedding_size) # (148,1,256)
	proj_matrix = proj_matrix.view(-1, ent_embedding_size, rel_embedding_size) # (148, 256,64)
	#print("After:",original.shape,proj_matrix.shape)
	result= torch.bmm(original,proj_matrix).view(-1, rel_embedding_size)   
	#print("Result:",result.shape)    
	return result

def projection_transR_pytorch_neg(original, proj_matrix):
	#print('NEG*Before:',original.shape,proj_matrix.shape) # (148,100) , (148,5000)
	ent_embedding_size = original.shape[0]
	rel_embedding_size = proj_matrix.shape[0] // ent_embedding_size
	original = original.view(-1,1,ent_embedding_size)
	proj_matrix = proj_matrix.view(-1, ent_embedding_size, rel_embedding_size)
	#print("NEG*After:",original.shape,proj_matrix.shape)
	result= torch.bmm(original,proj_matrix).view(-1, rel_embedding_size)   
	#print("NEG*Result:",result.shape)    
	return result



def projection_transD_pytorch_samesize(entity_embedding, entity_projection, relation_projection):
	return entity_embedding + torch.sum(entity_embedding * entity_projection, dim=1, keepdim=True) * relation_projection
