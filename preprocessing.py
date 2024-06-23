import datetime
from dateutil.parser import parse
from dateutil import parser
from datetime import timedelta
from collections import Counter
from collections import OrderedDict
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from graphutils import *


def ctime():
    return round(time.time() - stime, 5)


def neg(a, pos, all):
    while True:
        b = random.choice(all)
        if a + b not in check and b + a not in check:
            return b


L = ['NYC']  # ,'columbus','austin']
for Location in L:
    Location = Location
    print("Data Gen for", Location)
    NumtoDay = {6: 'Sun', 0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat'}

    DaytoNum = {'Sun': 0, 'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6}
    path = './' + Location + '/'
    is_sorted = False
    if is_sorted:
        print("Already sorted")
        f_s = open(path + Location + "_SCC_sorted.txt", 'r').readlines()
    else:
        f = open(path + Location + "_SCC.txt", 'r').readlines()
        f_s = sorted(f, key=lambda x: parse(x.strip().split('\t')[1]))
        f2 = open(path + Location + "_SCC_sorted.txt", 'w')
        s = ''
        for i in f_s:
            s += (i.strip() + '\n')
        f2.write(s[:-1])
        print("Sort and saved it to new file")

    f = open(path + Location + "_SCC_sorted.txt", 'r').readlines()
    f_s = f  # =sorted(f,key=lambda x: parse(x.strip().split('\t')[1]))
    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        location = {}
        alldata = ''
        alldata_geo = ''
        alldata_count = 0
        alldata_count_geo = 0

        uDict = OrderedDict()
        uCount = -1
        pDict = OrderedDict()
        pCount = 0
        pDict['PAD'] = 0
        cDict = OrderedDict()
        cCount = -1
        path2 = path[:-1] + '_' + str(int(r * 100)) + '/'
        print(path2)
        f_sorted = f_s[round(len(f_s) * (1 - r)):]
        print(r, f_sorted[0], f_sorted[-1])
        try:
            os.makedirs(path2)
        except:
            pCount = 0
        minlat = 180
        maxlat = -180
        minlon = 180
        maxlon = -180
        for i in f_sorted:
            # print(S)
            uid, date, pid, offset, lat, lon, cat = i.strip().split('\t')
            lat, lon = float(lat), float(lon)
            if lat > maxlat: maxlat = lat
            if lat < minlat: minlat = lat
            if lon > maxlon: maxlon = lon
            if lon < minlon: minlon = lon
        dlat = maxlat - minlat
        dlon = maxlon - minlon

        for i in f_sorted:
            uid, date, pid, offset, lat, lon, cat = i.strip().split('\t')
            lat, lon = float(lat), float(lon)
            X = int((lat - minlat) / dlat * 10)
            Y = int((lon - minlon) / dlon * 10)
            if X == 10: X = 9
            if Y == 10: Y = 9
            grid = X + Y * 10
            if pid in pDict:
                pid = pDict[pid]
            else:
                pCount += 1
                pDict[pid] = pCount
                pid = pDict[pid]
            location[pid] = (lat, lon)
            if uid in uDict:
                uid = uDict[uid]
            else:
                uCount += 1
                uDict[uid] = uCount
                uid = uDict[uid]
            if cat in cDict:
                cat = cDict[cat]
            else:
                cCount += 1
                cDict[cat] = cCount
                cat = cDict[cat]
            date = DTN(date, int(offset) // 60) % 168
            s = str(uid) + '\t' + str(date) + '\t' + str(pid) + '\t' + str(cat) + '\n'
            s2 = str(uid) + '\t' + str(date) + '\t' + str(pid) + '\t' + str(cat) + '\t' + str(grid) + '\n'
            alldata += s
            alldata_geo += s2
            alldata_count += 1
            alldata_count_geo += 1
        fa = open(path2 + "alldata2id.txt", 'w')
        fa.write(str(alldata_count) + '\n' + alldata)
        fa.close()
        fa2 = open(path2 + "alldata2id_geo.txt", 'w')
        fa2.write(str(alldata_count_geo) + '\n' + alldata_geo)
        fa2.close()
        f3 = open(path2 + "./user2id.txt", 'w')
        f3.write(str(len(uDict.keys())) + '\n')
        for i in uDict:
            f3.write(i + '\t' + str(uDict[i]) + '\n')
        f3.close()
        f4 = open(path2 + "./POI2id.txt", 'w')
        f4.write(str(len(pDict.keys())) + '\n')
        for i in pDict:
            f4.write(i + '\t' + str(pDict[i]) + '\n')
        f4.close()
        f5 = open(path2 + "./category2id.txt", 'w')
        f5.write(str(len(cDict.keys())) + '\n')
        for i in cDict:
            f5.write(i + '\t' + str(cDict[i]) + '\n')
        f5.close()

threshold = 10
f = open("friendship_new.txt").readlines()

for location in L:
    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        path = './' + location + '_' + str(int(r * 100)) + '/'
        print(path)
        c = 0
        stime = time.time()
        Yelp = True
        c = 0
        C = dict()
        for i in open(path + "user2id.txt").readlines()[1:]:
            a, b = i.strip().split('\t')
            C[a] = b
        pos = []
        check = {}
        for i in f:
            [a, b] = i.strip().split('\t')
            if a not in C or b not in C:
                continue
            if b + a not in check:
                pos.append([a, b])
                check[a + b] = True
                c += 1
                if c % 10000 == 0: print(c, ctime())

        random.shuffle(pos)
        val = round(len(pos) * 0.3)
        val_pos = pos[:val]
        test_pos = pos[val:]
        f7 = open(path + 'true_pairs_all.txt', 'w')
        f2 = open(path + 'true_pairs_val.txt', 'w')
        f3 = open(path + 'true_pairs_test.txt', 'w')
        f4 = open(path + 'false_pairs_all.txt', 'w')
        f5 = open(path + 'false_pairs_val.txt', 'w')
        f6 = open(path + 'false_pairs_test.txt', 'w')
        all = list(C.keys())
        stime = time.time()
        print("All start", ctime())
        for [a, b] in pos:
            s = str(C[a]) + '\t' + str(C[b]) + '\n'
            s_neg = str(C[a]) + '\t' + str(C[neg(a, pos, all)]) + '\n'
            f7.write(s)
            f4.write(s_neg)
        print("Val Start", ctime())
        for [a, b] in val_pos:
            s = str(C[a]) + '\t' + str(C[b]) + '\n'
            s_neg = str(C[a]) + '\t' + str(C[neg(a, val_pos, all)]) + '\n'
            f2.write(s)
            f5.write(s_neg)
        print("Test Start", ctime())
        for [a, b] in test_pos:
            s = str(C[a]) + '\t' + str(C[b]) + '\n'
            s_neg = str(C[a]) + '\t' + str(C[neg(a, test_pos, all)]) + '\n'
            f3.write(s)
            f6.write(s_neg)
        f2.close()
        f3.close()
        f4.close()
        f5.close()
        f6.close()
        f7.close()






