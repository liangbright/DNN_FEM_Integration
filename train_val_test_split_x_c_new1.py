import numpy as np
import glob

def train_val_test_split(folder, train_percent, n_samples=343, val_percent=0.1):
    #filelist_all=glob.glob(folder+"matMean/"+"*_i"+t+".pt")
    #filelist_all.sort()

    not_converged=48

    str_list=[]
    for n in range(0, n_samples):
        if n != not_converged:
            str_list.append(str(n))
    str_list.sort()

    filelist_all=[]
    for s in str_list:
        filelist_all.append(folder+'matMean/'+'p0_'+s+'_solid_matMean_p20_i90.pt')

    shape_idlist=[]
    for id in range(0, n_samples):
        for name in filelist_all:
            id_str="_"+str(id)+"_"
            if id_str in name:
                shape_idlist.append(id)
                break
    shape_idlist=np.array(shape_idlist)
    rng=np.random.RandomState(0)#seed=0
    rng.shuffle(shape_idlist)
    a=int(val_percent*len(shape_idlist))
    b=int((val_percent+train_percent)*len(shape_idlist))
    shape_idlist_val=shape_idlist[0:a]
    shape_idlist_train=shape_idlist[a:b]
    shape_idlist_test=shape_idlist[b:]
    filelist_train=[]
    for id in shape_idlist_train:
        for name in filelist_all:
            id_str="_"+str(id)+"_"
            if id_str in name:
                filelist_train.append(name)
                break
    filelist_train=np.array(filelist_train)
    filelist_val=[]
    for id in shape_idlist_val:
        for name in filelist_all:
            id_str="_"+str(id)+"_"
            if id_str in name:
                filelist_val.append(name)
                break
    filelist_val=np.array(filelist_val)
    filelist_test=[]
    for id in shape_idlist_test:
        for name in filelist_all:
            id_str="_"+str(id)+"_"
            if id_str in name:
                filelist_test.append(name)
                break
    filelist_test=np.array(filelist_test)

    if "4layer" in folder:
        # "remove" 47 - not converged
        shape_idlist_test=list(shape_idlist_test)
        del shape_idlist_test[-2]
        shape_idlist_test=np.array(shape_idlist_test)

        filelist_test=list(filelist_test)
        del filelist_test[-2]
        filelist_test=np.array(filelist_test)

    return filelist_train, filelist_val, filelist_test, shape_idlist_train, shape_idlist_val, shape_idlist_test
