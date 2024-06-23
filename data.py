

import os
import random
import numpy
from copy import deepcopy

from utils import LBSN
# Change the head of a triple randomly,
# without checking whether it is a false negative sample.

import torch
def negsample(distribution):
    # Select 기준 : k-1보다 크거나 같고 k보다 작으면 k-1 선택 (0과 1 사이면 0 선택이라 이말이지)
    # 근데 0이 추가되니까 3개라고 하면 [ 0.0 , 0.1, 0.5,  1.0] 에서 0과 1사이면 0 선택, 1과 2사이면 1선택, 2와 3사이면 2선택 해서 맞네
    x=random.random()
    s=0
    e=len(distribution)
    m=(s+e)//2
    while True:
        if distribution[m]>x: #Left
            if distribution[m-1]<=x:
                return m-1
            else:
                e=m
                m=(s+m)//2
            
        elif distribution[m]<=x: #Right
            if distribution[m+1]>x:
                return m
            else:
                s=m
                m=(m+e)//2
                
# Split the tripleList into #num_batches batches
# Note : num_batches 는 배치의 개수임 (사이즈가 아니고)
def getBatchList(LBSNList, num_batches):
    batchSize = len(LBSNList) // num_batches
    batchList = [0] * num_batches
    for i in range(num_batches - 1):
        batchList[i] = LBSNList[i * batchSize : (i + 1) * batchSize]
    batchList[num_batches - 1] = LBSNList[(num_batches - 1) * batchSize : ]
    return batchList


def getThreeElements(tripleList):
    headList = [triple.h for triple in tripleList]
    tailList = [triple.t for triple in tripleList]
    relList = [triple.r for triple in tripleList]
    return headList, tailList, relList

def getLBSNElements(LBSNList):
    userList = [lbsn.u for lbsn in LBSNList]
    timeList = [lbsn.t for lbsn in LBSNList]
    poiList = [lbsn.p for lbsn in LBSNList]
    catList=[lbsn.c for lbsn in LBSNList]
    try:
        geoList=[lbsn.g for lbsn in LBSNList]
        return userList, timeList, poiList, catList, geoList
    except:
        return userList, timeList, poiList,catList

def getFourElements_multiple(quadList):
    headList =[]
    tailList =[]
    relList =[]
    friendList=[]
    #quadList : 2d array [ [tensor, tensor, tensor], [] [] []]
    for sublist in quadList:
        h=[]
        t=[]
        r=[]
        f=[]
        for quad in sublist:
            h.append(quad.h)
            t.append(quad.t)
            r.append(quad.r)
            f.append(quad.f)
        headList.append(h)
        tailList.append(t)
        relList.append(r)
        friendList.append(f)
    return headList, tailList, relList,friendList

#make triple into quad
def makeQuad(oldTripleList,friendship):
    oldQuadList=[]
    for i in oldTripleList:
        h=i.h
        while True:
            f=random.sample(friendship[h].keys(),1)[0]
            if f in friendship[h] and friendship[h][f]>1:
                q=Quad(i.h,i.t,i.r,f)
                break
                #q=f+i
        oldQuadList.append(q)
    return oldQuadList



def corrupt_User(LBSNLIst, utpDict, distribution, NS):
    UList=[]
    for i in LBSNList:
        UList.append([])
        t=i.t
        p=i.p
        for _ in range(NS):
            while True:
                nu=negsample(distribution)
                if (nu,t,p) not in utpDict:
                    break
            UList[-1].append(nu)
    return UList

def corrupt_POI(LBSNList, utpDict,distribution,NS):
    NList=[]
    for i in LBSNList:
        NList.append([])
        u=i.u
        t=i.t
        p=i.p
        for _ in range(NS):
            while True:
                np=negsample(distribution)
                if (u,t,np) not in utpDict:
                    break
            NList[-1].append(np)
    return NList


def corrupt_POIorUser(LBSNList, utpDict, Udist, Pdist, NS, ratio):
    NList = []
    for i in LBSNList:
        NList.append([])
        u = i.u
        t = i.t
        p = i.p
        for _ in range(NS):
            x = random.random()
            if x > ratio:
                nu = u
                while True:
                    np = negsample(Pdist)
                    if (u, t, np) not in utpDict:
                        break
            else:
                np = p
                while True:
                    nu = negsample(Udist)
                    if (nu, t, p) not in utpDict:
                        break

            NList[-1].append([nu, t, np])
    return NList


def corrupt_POIorTimeorUserorCat(LBSNList, utpDict, ucpDict,Udist, Pdist, CattoPOI,NS, ratio):
    #ratio=0.3333
    NList = []
    for i in LBSNList:
        NList.append([])
        u = i.u
        t = i.t
        p = i.p
        c = i.c
        for _ in range(NS):
            x = random.random()
            if x > ratio:
                nu = u
                nt=t
                while True:
                    np = negsample(Pdist)
                    if (u, t, np) not in utpDict and (u,c,np) not in ucpDict:
                        break
            #elif x>2*ratio:
            #    nu=u
            #    np=p
            #    while True:
            #        nt=negsample(Tdist)
            #        if abs(nt-t)<6: continue
            #        if (nu,nt,np) not in utpDict:
            #            break
            else:
                np = p
                nt=t
                while True:
                    nu = negsample(Udist)
                    if (nu, t, p) not in utpDict and (nu,c,p) not in ucpDict:
                        break

            NList[-1].append([nu, nt, np,c])
    return NList


def getLBSNBatch_v2(LBSNList, batchSize, utpDict,Udist,Pdist,NumNS,ratio):
    LBSNList=random.sample(LBSNList,batchSize)
    pu,pt,pp,pc=getLBSNElements(LBSNList)
    n=corrupt_POIorUser(LBSNList,utpDict,Udist,Pdist,NumNS,ratio)
    np=[]
    nu=[]
    nt=[]
    for i in n:
        np.append([])
        nu.append([])
        nt.append([])
        for j in i:
            nu[-1].append(j[0])
            nt[-1].append(j[1])
            np[-1].append(j[2])

    #nc=corrupt_Category(LBSNList,utcDict,Cdist,NumNS)
    return pu, pt, pp,nu,nt,np


def corrupt_utcp(LBSNList, utpDict, ucpDict, Udist, Pdist, NS, ratio, POItoCat, flag):
    # ratio=0.3333
    if flag == 't':
        NList = []
        for i in LBSNList:
            NList.append([])
            u = i.u
            t = i.t
            p = i.p
            for _ in range(NS):
                x = random.random()
                if x > ratio:
                    nu = u
                    nt = t
                    while True:
                        np = negsample(Pdist)
                        if (u, t, np) not in utpDict:
                            break
                else:
                    np = p
                    nt = t
                    while True:
                        nu = negsample(Udist)
                        if (nu, t, p) not in utpDict:
                            break

                NList[-1].append([nu, nt, np])
    if flag == 'c':
        NList = []
        for i in LBSNList:
            NList.append([])
            u = i.u
            p = i.p
            c = i.c
            for _ in range(NS):
                x = random.random()
                if x > ratio:
                    nu = u
                    nc = c
                    while True:
                        np = negsample(Pdist)
                        nc = POItoCat[np]
                        if (nu, nc, np) not in ucpDict:
                            break
                else:
                    np = p
                    nc = c
                    while True:
                        nu = negsample(Udist)
                        if (nu, c, p) not in ucpDict:
                            break

                NList[-1].append([nu, nc, np])
    return NList





def getLBSNBatch_v3(LBSNList, batchSize, utpDict,ucpDict,Udist,Pdist,POItoCat,NumNS,ratio):
    LBSNList=random.sample(LBSNList,batchSize)
    pu,pt,pp,pc=getLBSNElements(LBSNList)
    n=corrupt_utcp(LBSNList,utpDict,ucpDict,Udist,Pdist,NumNS,ratio,POItoCat,'t')
    np=[]
    nu=[]
    nt=[]
    for i in n:
        np.append([])
        nu.append([])
        nt.append([])
        for j in i:
            nu[-1].append(j[0])
            nt[-1].append(j[1])
            np[-1].append(j[2])
    n=corrupt_utcp(LBSNList,utpDict,ucpDict,Udist,Pdist,NumNS,ratio,POItoCat,'c')
    nu2=[]
    nc=[]
    np2=[]
    for i in n:
        np2.append([])
        nu2.append([])
        nc.append([])
        for j in i:
            nu2[-1].append(j[0])
            nc[-1].append(j[1])
            np2[-1].append(j[2])
    #nc=corrupt_Category(LBSNList,utcDict,Cdist,NumNS)
    return pu, pt, pp,pc,nu,nt,np,nu2,nc,np2


def corrupt_utcpg(LBSNList, utpDict, ucpDict, ugpDict,Udist, Pdist, NS, ratio, POItoCat, POItoGrid,flag):
    # ratio=0.3333
    if flag == 't':
        NList = []
        for i in LBSNList:
            NList.append([])
            u = i.u
            t = i.t
            p = i.p
            for _ in range(NS):
                x = random.random()
                if x > ratio:
                    nu = u
                    nt = t
                    while True:
                        np = negsample(Pdist)
                        if (u, t, np) not in utpDict:
                            break
                else:
                    np = p
                    nt = t
                    while True:
                        nu = negsample(Udist)
                        if (nu, t, p) not in utpDict:
                            break

                NList[-1].append([nu, nt, np])
    if flag == 'c':
        NList = []
        for i in LBSNList:
            NList.append([])
            u = i.u
            p = i.p
            c = i.c
            for _ in range(NS):
                x = random.random()
                if x > ratio:
                    nu = u
                    nc = c
                    while True:
                        np = negsample(Pdist)
                        nc = POItoCat[np]
                        if (nu, nc, np) not in ucpDict:
                            break
                else:
                    np = p
                    nc = c
                    while True:
                        nu = negsample(Udist)
                        if (nu, c, p) not in ucpDict:
                            break

                NList[-1].append([nu, nc, np])

    if flag == 'g':
        NList = []
        for i in LBSNList:
            NList.append([])
            u = i.u
            p = i.p
            g = i.g
            for _ in range(NS):
                x = random.random()
                if x > ratio:
                    nu = u
                    ng = g
                    while True:
                        np = negsample(Pdist)
                        ng = POItoGrid[np]
                        if (nu, ng, np) not in ugpDict:
                            break
                else:
                    np = p
                    ng = g
                    while True:
                        nu = negsample(Udist)
                        if (nu, ng, np) not in ugpDict:
                            break

                NList[-1].append([nu, ng, np])
    return NList
def getLBSNBatch_Foursquare_GEO(LBSNList, batchSize, utpDict,ucpDict,ugpDict,Udist,Pdist,POItoCat,POItoGrid,NumNS,ratio):
    LBSNList=random.sample(LBSNList,batchSize)
    pu,pt,pp,pc,pg=getLBSNElements(LBSNList)
    n=corrupt_utcpg(LBSNList,utpDict,ucpDict,ugpDict,Udist,Pdist,NumNS,ratio,POItoCat,POItoGrid,'t')
    np=[]
    nu=[]
    nt=[]
    for i in n:
        np.append([])
        nu.append([])
        nt.append([])
        for j in i:
            nu[-1].append(j[0])
            nt[-1].append(j[1])
            np[-1].append(j[2])
    n=corrupt_utcpg(LBSNList,utpDict,ucpDict,ugpDict,Udist,Pdist,NumNS,ratio,POItoCat,POItoGrid,'c')
    nu2=[]
    nc=[]
    np2=[]
    for i in n:
        np2.append([])
        nu2.append([])
        nc.append([])
        for j in i:
            nu2[-1].append(j[0])
            nc[-1].append(j[1])
            np2[-1].append(j[2])
    #nc=corrupt_Category(LBSNList,utcDict,Cdist,NumNS)
    n=corrupt_utcpg(LBSNList,utpDict,ucpDict,ugpDict,Udist,Pdist,NumNS,ratio,POItoCat,POItoGrid,'g')
    nu3=[]
    ng=[]
    np3=[]
    for i in n:
        np3.append([])
        nu3.append([])
        ng.append([])
        for j in i:
            nu3[-1].append(j[0])
            ng[-1].append(j[1])
            np3[-1].append(j[2])
    #nc=corrupt_Category(LBSNList,utcDict,Cdist,NumNS)

    return pu, pt, pp,pc,pg,nu,nt,np,nu2,nc,np2,nu3,ng,np3

def corrupt_ucpg(LBSNList, ucpDict, ugpDict,Udist, Pdist, NS, ratio, POItoCat, POItoGrid,flag):
    # ratio=0.3333
    if flag == 'c':
        NList = []
        for i in LBSNList:
            NList.append([])
            u = i.u
            p = i.p
            c = i.c
            for _ in range(NS):
                x = random.random()
                if x > ratio:
                    nu = u
                    nc = c
                    while True:
                        np = negsample(Pdist)
                        nc = POItoCat[np]
                        if (nu, nc, np) not in ucpDict:
                            break
                else:
                    np = p
                    nc = c
                    while True:
                        nu = negsample(Udist)
                        if (nu, nc, np) not in ucpDict:
                            break

                NList[-1].append([nu, nc, np])

    if flag == 'g':
        NList = []
        for i in LBSNList:
            NList.append([])
            u = i.u
            p = i.p
            g = i.g
            for _ in range(NS):
                x = random.random()
                if x > ratio:
                    nu = u
                    ng = g
                    while True:
                        np = negsample(Pdist)
                        ng = POItoGrid[np]
                        if (nu, ng, np) not in ugpDict:
                            break
                else:
                    np = p
                    ng = g
                    while True:
                        nu = negsample(Udist)
                        if (nu, ng, np) not in ugpDict:
                            break

                NList[-1].append([nu, ng, np])
    return NList
def getLBSNBatch_Yelp_GEO(LBSNList, batchSize, ucpDict,ugpDict,Udist,Pdist,POItoCat,POItoGrid,NumNS,ratio):
    LBSNList=random.sample(LBSNList,batchSize)
    pu,pt,pp,pc,pg=getLBSNElements(LBSNList)
    n=corrupt_ucpg(LBSNList,ucpDict,ugpDict,Udist,Pdist,NumNS,ratio,POItoCat,POItoGrid,'c')
    nu=[]
    nc=[]
    np=[]
    for i in n:
        np.append([])
        nu.append([])
        nc.append([])
        for j in i:
            nu[-1].append(j[0])
            nc[-1].append(j[1])
            np[-1].append(j[2])
    #nc=corrupt_Category(LBSNList,utcDict,Cdist,NumNS)
    n=corrupt_ucpg(LBSNList,ucpDict,ugpDict,Udist,Pdist,NumNS,ratio,POItoCat,POItoGrid,'g')
    nu2=[]
    ng=[]
    np2=[]
    for i in n:
        np2.append([])
        nu2.append([])
        ng.append([])
        for j in i:
            nu2[-1].append(j[0])
            ng[-1].append(j[1])
            np2[-1].append(j[2])
    #nc=corrupt_Category(LBSNList,utcDict,Cdist,NumNS)

    return pu, pc,pp,pg,nu,nc,np,nu2,ng,np2


def corrupt_utpg(LBSNList, utpDict, ugpDict,Udist, Pdist, NS, ratio, POItoGrid,flag):
    # ratio=0.3333
    if flag == 't':
        NList = []
        for i in LBSNList:
            NList.append([])
            u = i.u
            t = i.t
            p = i.p
            for _ in range(NS):
                x = random.random()
                if x > ratio:
                    nu = u
                    nt = t
                    while True:
                        np = negsample(Pdist)
                        if (u, t, np) not in utpDict:
                            break
                else:
                    np = p
                    nt = t
                    while True:
                        nu = negsample(Udist)
                        if (nu, t, p) not in utpDict:
                            break

                NList[-1].append([nu, nt, np])
    if flag == 'g':
        NList = []
        for i in LBSNList:
            NList.append([])
            u = i.u
            p = i.p
            g = i.g
            for _ in range(NS):
                x = random.random()
                if x > ratio:
                    nu = u
                    ng = g
                    while True:
                        np = negsample(Pdist)
                        ng = POItoGrid[np]
                        if (nu, ng, np) not in ugpDict:
                            break
                else:
                    np = p
                    ng = g
                    while True:
                        nu = negsample(Udist)
                        if (nu, ng, np) not in ugpDict:
                            break

                NList[-1].append([nu, ng, np])
    return NList
def getLBSNBatch_Others_GEO(LBSNList, batchSize, utpDict,ugpDict,Udist,Pdist,POItoGrid,NumNS,ratio):
    LBSNList=random.sample(LBSNList,batchSize)
    pu,pt,pp,pc,pg=getLBSNElements(LBSNList)
    n=corrupt_utpg(LBSNList,utpDict,ugpDict,Udist,Pdist,NumNS,ratio,POItoGrid,'t')
    np=[]
    nu=[]
    nt=[]
    for i in n:
        np.append([])
        nu.append([])
        nt.append([])
        for j in i:
            nu[-1].append(j[0])
            nt[-1].append(j[1])
            np[-1].append(j[2])
    n=corrupt_utpg(LBSNList,utpDict,ugpDict,Udist,Pdist,NumNS,ratio,POItoGrid,'g')
    nu2=[]
    ng=[]
    np2=[]
    for i in n:
        np2.append([])
        nu2.append([])
        ng.append([])
        for j in i:
            nu2[-1].append(j[0])
            ng[-1].append(j[1])
            np2[-1].append(j[2])
    #nc=corrupt_Category(LBSNList,utcDict,Cdist,NumNS)

    return pu, pt, pp,pg,nu,nt,np,nu2,ng,np2
