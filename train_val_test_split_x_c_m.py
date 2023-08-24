import numpy as np
import torch
def train_val_test_split(path, train_percent=[0.5, 0.8], val_percent=[0.1, 0.1], seed=0):
    #p0_0_solid_matMean_p20_i0
    #path  ??/data/343c1.5_125mat/

    rng=np.random.RandomState(seed)

    shape_idlist=[]
    for n in range(0, 343):
        if n !=48:
            shape_idlist.append(n)
    shape_idlist=np.array(shape_idlist)
    rng.shuffle(shape_idlist)
    a=int(val_percent[0]*len(shape_idlist))
    b=int((val_percent[0]+train_percent[0])*len(shape_idlist))
    shape_idlist_val=shape_idlist[0:a]
    shape_idlist_train=shape_idlist[a:b]
    shape_idlist_test=shape_idlist[b:]

    mat_idlist=[]
    for n in range(0, 125):
        mat_idlist.append(n)
    mat_idlist=np.array(mat_idlist)
    rng.shuffle(mat_idlist)
    a=int(val_percent[1]*len(mat_idlist))
    b=int((val_percent[1]+train_percent[1])*len(mat_idlist))
    mat_idlist_val=mat_idlist[0:a]
    mat_idlist_train=mat_idlist[a:b]
    mat_idlist_test=mat_idlist[b:]

    filelist_train_p0=[]
    for n in shape_idlist_train:
        folder=path
        filelist_train_p0.append(folder+'p0_'+str(n)+'_solid')

    filelist_train=[]
    for n in shape_idlist_train:
        for m in mat_idlist_train:
            folder=path+'mat'+str(m)+'/'
            filelist_train.append([folder+'p0_'+str(n)+'_solid_mat'+str(m)+'_p20_i0',
                                   folder+'p0_'+str(n)+'_solid_mat'+str(m)+'_p20_i18',
                                   m])

    filelist_val=[]
    for n in shape_idlist_val:
        for m in mat_idlist_val:
            folder=path+'mat'+str(m)+'/'
            filelist_val.append([folder+'p0_'+str(n)+'_solid_mat'+str(m)+'_p20_i0',
                                 folder+'p0_'+str(n)+'_solid_mat'+str(m)+'_p20_i18',
                                 m])

    filelist_test=[]
    for n in shape_idlist_test:
        for m in mat_idlist_test:
            folder=path+'mat'+str(m)+'/'
            filelist_test.append([folder+'p0_'+str(n)+'_solid_mat'+str(m)+'_p20_i0',
                                  folder+'p0_'+str(n)+'_solid_mat'+str(m)+'_p20_i18',
                                  m])

    return filelist_train_p0, filelist_train, filelist_val, filelist_test


