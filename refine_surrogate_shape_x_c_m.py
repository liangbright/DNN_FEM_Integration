#%%
net_list=["NetXCM1A_Encoder3('BaseNet0',3,128,2,1,1,1,5)_Net1('BaseNet5b',3,10,256,4,1,1,1,3,'softplus')_1"
         ]
#%%
import os
import torch
from train_val_test_split_x_c_m import train_val_test_split
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--folder_data', default='./data/343c1.5_125mat/', type=str)
parser.add_argument('--folder_result', default='./result/forward/', type=str)
parser.add_argument('--train_percent', default=0.5, type=float)
parser.add_argument('--loss1', default=0.01, type=float)
parser.add_argument('--Rmax', default=0.005, type=float)
parser.add_argument('--warm_up_T0', default=10, type=int)
parser.add_argument('--max_iter2', default=1000, type=int)
arg = parser.parse_args()
print(arg)
#%%
(filelist_train_p0,
 filelist_train,
 filelist_val,
 filelist_test)=train_val_test_split(arg.folder_data)

all_mat=torch.load('./data/125mat.pt')['mat_str']
#%%
for net in net_list:
    for k in range(0, len(filelist_test)):
        mesh_p0=filelist_test[k][0]
        mesh_px=filelist_test[k][1]
        mat_id=filelist_test[k][2]
        mat=all_mat[mat_id]
        mesh_px_init=arg.folder_result+net+'/test/pred_'+mesh_px.split("/")[-1]
        mesh_px_pred=mesh_px_init+'_refine_R1'
        #'''
        cmd=("python aorta_FEA_C3D8_SRI_R1.py"
             +" --cuda "+str(arg.cuda)
             +" --mesh_p0 "+mesh_p0
             +" --mesh_px_init "+mesh_px_init
             +" --mesh_px_pred "+mesh_px_pred
             +" --pressure 18"
             +" --mat " +'"'+mat+'"'
             +" --loss1 "+str(arg.loss1)
             +" --Rmax "+str(arg.Rmax)
             +" --max_iter2 "+str(arg.max_iter2)
             +" --warm_up_T0 "+str(arg.warm_up_T0)
             +" --save_by_vtk False"
             )
        print(cmd)
        os.system(cmd)
        #'''
        #break
