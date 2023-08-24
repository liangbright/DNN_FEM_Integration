import pandas as pd
import matplotlib.pyplot as plt
from analyze_surrogate_shape_x_c_m_functions import get_data_frame, get_result, get_table, get_filelist
#%%
pd.set_option("display.max_columns", None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)
#%%
folder_data='./data/343c1.5_125mat/'
folder_result="./result/forward/"
stress='VM'
#%%
Net1="NetXCM1A_Encoder3('BaseNet0',3,128,2,1,1,1,5)_Net1('BaseNet5b',3,10,256,4,1,1,1,3,'softplus')_1"
#%%
Net1_result_test=get_data_frame(Net1, 'Net1', folder_data, folder_result, test_or_val='test',
                                stress=stress, refine=False)
print(Net1_result_test)
#%%
Net1_result_test_r=get_data_frame(Net1, 'Net1', folder_data, folder_result, test_or_val='test',
                                  stress=stress, refine=True, iter_threshold=150)
print(Net1_result_test_r)
print("error%", str(100*Net1_result_test_r['error'].values[0]/1781)+'%')
#%%
Net1_result_test.to_csv(folder_result+'x_c_m_result_test.csv')
Net1_result_test_r.to_csv(folder_result+'x_c_m_result_test_r_R1.csv')
#%%
result=get_result(Net1,
                  folder_data, folder_result, test_or_val='test', stress='VM', refine=False, iter_threshold=None)
mrse_mean, mrse_max, MAPE_list, APE_list, time_cost, filelist_true, filelist_pred = result
#%%
plt.hist(MAPE_list[MAPE_list<=1], bins=100)
#%%
plt.hist(APE_list[APE_list<=1], bins=100)
#%%
import numpy as np
print(np.sum(MAPE_list>0.1)/len(MAPE_list))
print(np.sum(APE_list>0.1)/len(APE_list))

print(np.sum(MAPE_list>0.05)/len(MAPE_list))
print(np.sum(APE_list>0.05)/len(APE_list))
