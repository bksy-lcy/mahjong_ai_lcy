# -*- coding: utf-8 -*-

import os
import bz2
import hashlib
import sqlite3
import threading
from datetime import datetime
import numpy as np

import requests

db_folder = os.path.dirname(os.path.realpath(__file__))

def get_rb(opt):
    ls = opt.split()
    for st in ls:
        if st[0:4]=='seed' :
            ss = st[6:-1].split(',')
            return int(ss[0]),int(ss[1])

def get_score(opt):
    ls = opt.split()
    for st in ls:
        if st[0:2]=='sc' :
            ss = st[4:-1].split(',')
            return [int(ss[0])+int(ss[1]),int(ss[2])+int(ss[3]),int(ss[4])+int(ss[5]),int(ss[6])+int(ss[7])]
        
def get_tar(opt):
    ls = opt.split()
    for st in ls:
        if st[0:5]=='owari' :
            ss = st[7:-1].split(',')
            return [float(ss[1])/100,float(ss[3])/100,float(ss[5])/100,float(ss[7])/100]

def rank_tar(target):
    # print(target)
    dat=[0,0,0,0]
    pt=[0.75,0.25,0,-1]
    for i in range(4):
        for j in range(4):
            if j<i and target[j]>=target[i] :
                dat[i] +=1
            if j>i and target[j]>target[i] :
                dat[i] +=1
        dat[i]=pt[dat[i]]
    return dat
        
def get_data(binary_content):
    content = str(binary_content,'utf-8')
    # print(content)
    opts = content.split('>')
    datas = []
    target = [0,0,0,0]
    wind = 0
    oya = 0
    ba = 0
    for opt in opts:
        if opt[0:5]=='<INIT' :
            roundnumber,ba = get_rb(opt)
            wind = roundnumber // 4
            oya = roundnumber % 4
        if opt[0:6]=='<AGARI':
            score = get_score(opt)
            datas.append((wind,oya,ba,score[0],score[1],score[2],score[3]))
            if  opt.find('owari')>=0 :
                target = get_tar(opt)
    return datas, target

def main():
    db_file = os.path.join(db_folder, "2020.db")
    train_dat = []
    connection = sqlite3.connect(db_file)
    with connection:
        cursor = connection.cursor()
        # cursor.execute("SELECT log_content from logs where is_processed = 1 and was_error = 0 limit 0,1;")
        cursor.execute("SELECT log_content from logs where is_processed = 1 and was_error = 0;")
        data = cursor.fetchall()
        cnt = 1
        for log in data:
            try:
                binary_content = bz2.decompress(log[0])
            except:
                print("Can not uncompress log content")
            # print(binary_content)
            # print(get_data(binary_content))
            datas, target = get_data(binary_content)
            if target[0]<0.005 and target[1]<0.005 and target[2]<0.005 and target[3]<0.005 :
                continue;
            target=rank_tar(datas[-1][3:])
            # target = tuple(target)
            for dat in datas:
                train_dat.append([list(dat),target])
            if cnt%1000==0 :
                print(cnt)
            cnt += 1
    # print(train_dat[0:])
    train_dat=np.array(train_dat)
    np.save('train_dat',train_dat)
    
if __name__ == "__main__":
    main()