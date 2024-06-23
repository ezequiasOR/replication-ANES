

import os
from copy import deepcopy
import pickle
import random
import numpy as np
import time
import datetime
from itertools import groupby

import loss

# LBSN Instance (user, time, poi, category)

class LBSN(object):
	def __init__(self, user, time, poi,category,grid):
		self.u = user
		self.t = time
		self.p = poi
		self.c = category
		self.g = grid
	def info(self):
		return [self.u,self.t,self.p,self.c,self.g]
# Compare two Triples in the order of head, relation and tail
def cmp_head(a, b):
	return (a.h < b.h or (a.h == b.h and a.r < b.r) or (a.h == b.h and a.r == b.r and a.t < b.t))

# Compare two Triples in the order of tail, relation and head
def cmp_tail(a, b):
	return (a.t < b.t or (a.t == b.t and a.r < b.r) or (a.t == b.t and a.r == b.r and a.h < b.h))

# Compare two Triples in the order of relation, head and tail
def cmp_rel(a, b):
	return (a.r < b.r or (a.r == b.r and a.h < b.h) or (a.r == b.r and a.h == b.h and a.t < b.t))

def minimal(a, b):
	if a > b:
		return b
	return a

def cmp_list(a, b):
	return (minimal(a.h, a.t) > minimal(b.h, b.t))

# Write a list of Triples into a file, with three numbers (head tail relation) per line
def process_list(tripleList, dataset, filename):
	with open(os.path.join('./datasets/', dataset, filename), 'w') as fw:
		fw.write(str(len(tripleList)) + '\n')
		for triple in tripleList:
			fw.write(str(triple.h) + '\t' + str(triple.t) + '\t' + str(triple.r) + '\n')

emptyLBSN = LBSN(0, 0, 0, 0,0)

def getRel(triple):
	return triple.r

# Gets the number of entities/relations/triples
def getAnythingTotal(inPath, fileName):
	with open(os.path.join(inPath, fileName), 'r') as fr:
		for line in fr:
			return int(line)
def newt(t):
    day = t // 24
    hour = t % 24
    nt = day * 4 + hour // 6
    return nt

def loadLBSN(inPath, fileName,dtype='Foursquare'):
	with open(os.path.join(inPath, fileName), 'r') as fr:
		i = 0
		LBSNList = []
		for line in fr:
			if i == 0:
				LBSNTotal = int(line)
				i += 1
			else:
				if dtype=='Foursquare':
					line_split = line.split('\t')
					user = int(line_split[0])
					time = int(line_split[1])
					poi = int(line_split[2])
					try:
						category = int(line_split[3])
					except:
						category=0
					try:
						grid=int(line_split[4])
					except:
						grid=0
					LBSNList.append(LBSN(user,time,poi,category,grid))
				elif dtype=='Yelp':
					time=0
					line_split = line.split('\t')
					user = int(line_split[0])
					category = int(line_split[1])
					poi = int(line_split[2])
					try:
						grid=int(line_split[3])
					except:
						grid=0
					LBSNList.append(LBSN(user,time,poi,category,grid))
				else:
					category=0
					line_split = line.split('\t')
					user = int(line_split[0])
					time = int(line_split[1])
					poi = int(line_split[2])
					try:
						grid=int(line_split[3])
					except:
						grid=0
					LBSNList.append(LBSN(user,time,poi,category,grid))


	LBSNDict = {}
	for LBSNelem in LBSNList:
		LBSNDict[(LBSNelem.u, LBSNelem.t, LBSNelem.p, LBSNelem.c)] = True

	return LBSNTotal, LBSNList, LBSNDict

# Calculate the statistics of datasets
def calculate_one_or_many(dataset):
	tripleTotal, tripleList, tripleDict = loadTriple('./datasets/' + dataset, 'triple2id.txt')
	# You should sort first before groupby!
	tripleList.sort(key=lambda x: (x.r, x.h, x.t))
	grouped = [(k, list(g)) for k, g in groupby(tripleList, key=getRel)]
	num_of_relations = len(grouped)
	head_per_tail_list = [0] * num_of_relations
	tail_per_head_list = [0] * num_of_relations

	one_to_one = []
	one_to_many = []
	many_to_one = []
	many_to_many = []

	for elem in grouped:
	    headList = []
	    tailList = []
	    for triple in elem[1]:
	        headList.append(triple.h)
	        tailList.append(triple.t)
	    headSet = set(headList)
	    tailSet = set(tailList)
	    head_per_tail = len(headList) / len(tailSet)
	    tail_per_head = len(tailList) / len(headSet)
	    head_per_tail_list[elem[0]] = head_per_tail
	    tail_per_head_list[elem[0]] = tail_per_head
	    if head_per_tail < 1.5 and tail_per_head < 1.5:
	        one_to_one.append(elem[0])
	    elif head_per_tail >= 1.5 and tail_per_head < 1.5:
	        many_to_one.append(elem[0])
	    elif head_per_tail < 1.5 and tail_per_head >= 1.5:
	        one_to_many.append(elem[0])
	    else:
	        many_to_many.append(elem[0])

	# Classify test triples according to the type of relation
	testTotal, testList, testDict = loadTriple('./datasets/' + dataset, 'test2id.txt')
	testList.sort(key=lambda x: (x.r, x.h, x.t))
	test_grouped = [(k, list(g)) for k, g in groupby(testList, key=getRel)]

	one_to_one_list = []
	one_to_many_list = []
	many_to_one_list = []
	many_to_many_list = []

	for elem in test_grouped:
	    if elem[0] in one_to_one:
	        one_to_one_list.append(elem[1])
	    elif elem[0] in one_to_many:
	        one_to_many_list.append(elem[1])
	    elif elem[0] in many_to_one:
	        many_to_one_list.append(elem[1])
	    else:
	        many_to_many_list.append(elem[1])

	one_to_one_list = [x for j in one_to_one_list for x in j]
	one_to_many_list = [x for j in one_to_many_list for x in j]
	many_to_one_list = [x for j in many_to_one_list for x in j]
	many_to_many_list = [x for j in many_to_many_list for x in j]

	process_list(one_to_one_list, dataset, 'one_to_one_test.txt')
	process_list(one_to_many_list, dataset, 'one_to_many_test.txt')
	process_list(many_to_one_list, dataset, 'many_to_one_test.txt')
	process_list(many_to_many_list, dataset, 'many_to_many_test.txt')

	with open(os.path.join('./datasets/', dataset, 'head_tail_proportion.pkl'), 'wb') as fw:
		pickle.dump(tail_per_head_list, fw)
		pickle.dump(head_per_tail_list, fw)

def which_loss_type(num):
	if num == 0:
		return loss.marginLoss
	elif num == 1:
		return loss.EMLoss
	elif num == 2:
		return loss.WGANLoss
	elif num == 3:
		return nn.MSELoss