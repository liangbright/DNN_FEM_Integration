import sys
sys.path.append("./mesh")
import torch
from QuadMesh import QuadMesh
from PolyhedronMesh import PolyhedronMesh
#%%
def get_boundary0(n_layers, N1=50, N2=5000):
    boundary0=[]
    for n in range(0, n_layers+1):
        boundary0.append(torch.arange(0, N1, 1)+n*N2)
    boundary0=torch.cat(boundary0, dim=0)
    return boundary0

def get_boundary1(n_layers, N1=50, N2=5000):
    boundary1=[]
    for n in range(0, n_layers+1):
        boundary1.append(torch.arange(N2-N1, N2, 1)+n*N2)
    boundary1=torch.cat(boundary1, dim=0)
    return boundary1

def get_solid_mesh_cfg(filename_shell, n_layers=1, N1=50):
    aorta_shell=QuadMesh()
    #aorta_shell.load_from_vtk(filename_shell, dtype=torch.float64)
    aorta_shell.load_from_torch(filename_shell)
    N2 = aorta_shell.node.shape[0]
    element_surface_pressure=aorta_shell.element
    element_surface_free=aorta_shell.element+N2*n_layers
    boundary0=get_boundary0(n_layers, N1, N2)
    boundary1=get_boundary1(n_layers, N1, N2)
    return boundary0, boundary1, element_surface_pressure, element_surface_free

def cal_u_boundary(node, boundary, r):
    pos=node[boundary]
    center=pos.mean(dim=1, keepdim=True)
    direction=pos-center
    direction=direction/torch.norm(direction, p=2, dim=1, keepdim=True)
    r=r.view(-1,1)
    u=r*direction
    return u

def shell_to_solid(shell_node, shell_element, thickness):
    #four layers
    #thickness: [0.5, 0.5, 0.5, 0.5] four layers, 0.5mmm thickness per layer
    normal=QuadMesh.cal_node_normal(shell_node, shell_element)
    node=[]
    node.append(shell_node)
    for n in range(0, len(thickness)):
        node.append(shell_node+sum(thickness[0:(n+1)])*normal)
    node=torch.cat(node, dim=0)
    element=[]
    for n in range(0, len(thickness)):
        for m in range(0, shell_element.shape[0]):
            e=list(shell_element[m]+n*shell_node.shape[0])
            e.extend(list(shell_element[m]+(n+1)*shell_node.shape[0]))
            element.append(e)
    element=torch.tensor(element, dtype=torch.int64)
    return node, element
#%%
if __name__ == '__main__':
    #%%
    Mesh_X=PolyhedronMesh()
    Mesh_X.load_from_vtk('../../../MLFEA/TAA/bav17_P0_best_solid_1layers.vtk', torch.float64)
    Mesh_X.save_by_torch('../../../MLFEA/TAA/bav17_P0_best_solid_1layers.pt')
    #%%
    shell_mesh=QuadMesh()
    P0_name="bav17_AortaModel_P0_best"
    #P0_name="tube90"
    shell_mesh.load_from_vtk("../../../MLFEA/TAA/"+P0_name+".vtk", dtype=torch.float64)
    thickness=[0.5, 0.5, 0.5, 0.5]
    node, element=shell_to_solid(shell_mesh.node, shell_mesh.element, thickness)
    solid_mesh=PolyhedronMesh()
    solid_mesh.node=node
    solid_mesh.element=element
    solid_mesh.save_by_vtk("../../../MLFEA/TAA/"+P0_name+"_solid_"+str(len(thickness))+"layers.vtk")
    #%%





