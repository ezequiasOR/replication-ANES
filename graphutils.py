import datetime
from dateutil.parser import parse
from datetime import timedelta
from dateutil import parser
import time
DaytoNum ={'Sun':0,'Mon':1,'Tue':2,'Wed':3,'Thu':4,'Fri':5,'Sat':6}
def ctime(stime):
    return round(time.time()-stime,5)
def timedelta2int(td):
    res = round(td.microseconds/float(1000000)) + (td.seconds + td.days * 24 * 3600)
    return res

def time_diff(t1,o1,t2,o2):
    time1=parser.parse(t1)+timedelta(minutes=int(o1))
    time2=parser.parse(t2)+timedelta(minutes=int(o2))
    return abs(timedelta2int(time1-time2))
def realtime(t1,o1):
    return str(parser.parse(t1)+timedelta(minutes=int(o1)))[:-6]
def DTN(s,offset):
    try:
        day,_,hour=s.split(' ')[0:3]
        timeslot = (DaytoNum[day] * 24 + int(hour) + offset)%168
    except: #AUSTIN : Monday is 0 (<> Foursquare : Sunday is 0)
        day=(parser.parse(s).weekday()+1)%7
        hour=parser.parse(s).hour
        timeslot=(day*24+int(hour)+offset)%168
    if timeslot<0: timeslot+=168
    return timeslot




def l2t(l):
    return l.split('\t')[1]
def split_by_week(log,boundary):
    res=[]
    check={}
    for i in range(len(boundary)-1): res.append([])
    for i in range(len(log)): check[i]=0
    s=0
    e=len(boundary)-1
    st=parser.parse(l2t(log[0]))
    et=parser.parse(l2t(log[-1]))
    while boundary[s]<st:
        s+=1
    while boundary[e]>et:
        e-=1
    s-=1
    e+=1 # boundary 길이 6개, 총 구간 5개 [0,1,2,3,4]
    for j in range(s,e):
        for k in range(len(log)):
            t=parser.parse(l2t(log[k]))
            if check[k]==0 and t>=boundary[j] and t<boundary[j+1]:
                res[j].append(log[k])
                check[k]=1
    return res
def unique(log):
    u={}
    for i in log:
        u[i.split('\t')[2]]=1
    return len(u)
def activity(log):
    a={}
    for i in log:
        a[i.split('\t')[3]]=1
    return len(a)

def Co_occurence(logA, logB):
    cooc=0 # co-occurence
    for i in logA:
        if i==[]: continue
        for j in logB:
            if j==[]: continue
            for k in i:
                try:
                    _,t,p,c,o=k.strip().split('\t')
                except:
                    _,t,p=k.strip().split('\t')
                for l in j:
                    try:
                        _,tt,pp,cc,oo=l.strip().split('\t')
                    except:
                        _,tt,pp=l.strip().split('\t')
                    if pp==p: # Found overlap
                        ## 시간 제약 둘거면 여기다 넣으면 됨
                        cooc+=1
                        break
    return cooc
def Co_occurence_deepwalk(logA, logB):
    cooc=0 # co-occurence
    for i in logA:
        if i==[]: continue
        for j in logB:
            if j==[]: continue
            for k in i:
                try:
                    _,t,p,c,o=k.strip().split('\t')
                except:
                    _,t,p=k.strip().split('\t')
                for l in j:
                    try:
                        _,tt,pp,cc,oo=l.strip().split('\t')
                    except:
                        _,tt,pp=l.strip().split('\t')
                    if pp==p: # Found overlap
                        ## 시간 제약 둘거면 여기다 넣으면 됨
                        cooc+=1
                        return cooc
                        break
    return cooc
def POI_cooc(logA,Graph): # Overlapped POIs in a single log
    overlap=[]
    for i in logA:
        if len(i)<2: continue
        for j in range(len(i)):
            try:
                _, t, p, c, o = i[j].strip().split('\t')
            except:
                _,t,p=i[j].strip().split('\t')
            p=int(p)
            for k in range(j+1,len(i)):
                try:
                    _, tt, pp, cc, oo = i[k].strip().split('\t')
                except:
                    _,tt,pp=i[k].strip().split('\t')
                pp=int(pp)
                if p!=pp: # Self-loop 제거? 맞나? 아니면 걍 빼면되고
                    overlap.append([p,pp])
    for [p,pp] in overlap:
        if pp not in Graph[p]: Graph[p][pp]=0
        if p not in Graph[pp]: Graph[pp][p]=0
        Graph[p][pp]+=1
        Graph[pp][p]+=1 # Self-Loop 있으면 반 나눠야겠지
    return Graph
def divide_trajectory(traj,interval,u,data_type):
    sentences=[]
    for i in range(len(traj)):
        head=i-1
        rear=i+1
        mtime=parser.parse(traj[i].split('\t')[1])
        sentences.append([u])
        while head>=0:
            htime=parser.parse(traj[head].split('\t')[1])
            if interval<mtime-htime:
                break
            head-=1
        head+=1
        while rear<=len(traj)-1:
            rtime=parser.parse(traj[rear].split('\t')[1])
            if interval<rtime-mtime:
                break
            rear+=1
        rear-=1
        for t in traj[head:rear+1]:
            t=t.strip()
            if data_type=='POI':
                sentences[-1].append(t.split('\t')[2])
            else:
                sentences[-1].append(t.split('\t')[3])
    return sentences







def getBatchList(LBSNList, batchSize,YList):
    num_batches= len(LBSNList) // batchSize
    batchList = [0] * num_batches
    batchY=[0]*num_batches
    for i in range(num_batches - 1):
        batchList[i] = LBSNList[i * batchSize : (i + 1) * batchSize]
        batchY[i]=YList[i * batchSize : (i + 1) * batchSize]
    batchList[num_batches - 1] = LBSNList[(num_batches - 1) * batchSize : num_batches*batchSize]
    batchY[num_batches-1]=YList[(num_batches-1) * batchSize : num_batches*batchSize]
    return batchList,batchY