#%%
temp_list=["Encoder3('BaseNet0',3,128,2,1,1,1,3)_Net1('BaseNet5b',3,3,512,4,1,1,1,3,'softplus')_float32",
           "Linear_encoder(30000,3,0.1)_MLP1b(3,512,4,30000)_float32",
           "MeshGraphNet(3,8,5,128,3,0.1)_float32",
           "TransEncoder4(2,256,2,16)_TransDecoder4()_float32"
           "UNet(512)_TransDecoder4()_float32"
           ]
#%%
import os
from train_val_test_split_x_c_new1 import train_val_test_split
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--path', default='D:/MLFEA/TAA/', type=str)#modify it
parser.add_argument('--folder_data', default='data/343c1.5_fast/', type=str)
parser.add_argument('--folder_result', default='result/forward/', type=str)
parser.add_argument('--train_percent', default=0.5, type=float)
parser.add_argument('--loss1', default=0.01, type=float)
parser.add_argument('--Rmax', default=0.005, type=float)
parser.add_argument('--warm_up_T0', default=10, type=int)
parser.add_argument('--max_iter2', default=1000, type=int)
arg = parser.parse_args()
print(arg)
#%%
percent=arg.train_percent
folder_data=arg.path+arg.folder_data
for net in temp_list:
    (filelist_train, filelist_val, filelist_test,
     shape_idlist_train, shape_idlist_val, shape_idlist_test)=train_val_test_split(folder_data, percent)
    folder_result=arg.path+arg.folder_result+net+'/'+str(percent)+"/matMean/test/"
    if os.path.exists(folder_result) == False:
        os.makedirs(folder_result)
    for k in shape_idlist_test:
        mesh_p0='p0_'+str(k)+'_solid'
        mesh_px_pred=mesh_p0+'_matMean_p18_pred_refine_R1'
        mesh_px_init=mesh_p0+'_matMean_p18_pred'
        cmd=("python aorta_FEA_C3D8_SRI_R1.py"
             +" --cuda "+str(arg.cuda)
             +" --mesh_p0 "+folder_data+mesh_p0
             +" --mesh_px_pred "+folder_result+mesh_px_pred
             +" --mesh_px_init "+folder_result+mesh_px_init
             +" --pressure 18"
             +" --loss1 "+str(arg.loss1)
             +" --Rmax "+str(arg.Rmax)
             +" --max_iter2 "+str(arg.max_iter2)
             +" --warm_up_T0 "+str(arg.warm_up_T0)
             +" --save_by_vtk False"
             )
        print(cmd)
        os.system(cmd)
