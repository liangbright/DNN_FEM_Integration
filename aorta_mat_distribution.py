import sys
sys.path.append("./mesh")

import torch
from PolyhedronMesh import PolyhedronMesh

def generate_mat_distribution(id, folder='./data/'):
    if id == 0:
        return generate_mat_distribution0(folder)
    elif id == 1:
        return generate_mat_distribution1(folder)
    elif id == 2:
        return generate_mat_distribution2(folder)
    elif id == 3:
        return generate_mat_distribution3(folder)

def generate_mat_distribution0(folder='./data/'):
    mesh_tube=PolyhedronMesh()
    mesh_tube.load_from_torch(folder+'aorta_tube_solid_1layers.pt')
    matMean=torch.load(folder+'125mat.pt')['mean_mat']
    Mat=torch.tensor(matMean, dtype=torch.float64)
    Mat=Mat.view(-1,6).expand(mesh_tube.element.shape[0], 6)
    return Mat

def generate_mat_distribution1(folder='./data/'):
    mesh_tube=PolyhedronMesh()
    mesh_tube.load_from_torch(folder+'aorta_tube_solid_1layers.pt')
    node=mesh_tube.node
    element=mesh_tube.element
    matMean=torch.load(folder++'125mat.pt')['mean_mat']
    #element center
    center=node[element].mean(dim=1)
    z=center[:,2]
    Mat=torch.zeros((element.shape[0], 6), dtype=torch.float64)
    Mat[:,0]=matMean[0]*(1+0.1*torch.sin(0.01*z)+0.1*torch.sin(0.1*z)+0.1*torch.sin(z))
    Mat[:,1]=matMean[1]*(1+0.1*torch.sin(0.009*z)+0.1*torch.sin(0.09*z)+0.1*torch.sin(z))
    Mat[:,2]=matMean[2]*(1+0.1*torch.sin(0.008*z)+0.1*torch.sin(0.08*z)+0.1*torch.sin(z))
    Mat[:,3]=matMean[3]*(1+0.1*torch.sin(0.007*z)+0.1*torch.sin(0.07*z)+0.1*torch.sin(z))
    Mat[:,4]=matMean[4] # mean orientation
    Mat[:,5]=matMean[5] # 1e5, a known constant
    return Mat

def generate_mat_distribution2(folder='./data/'):
    mesh_tube=PolyhedronMesh()
    mesh_tube.load_from_torch(folder+'aorta_tube_solid_1layers.pt')
    node=mesh_tube.node
    element=mesh_tube.element
    matMean=torch.load(folder+'125mat.pt')['mean_mat']
    #element center
    center=node[element].mean(dim=1)
    z=center[:,2]*10
    Mat=torch.zeros((element.shape[0], 6), dtype=torch.float64)
    Mat[:,0]=matMean[0]*(1+0.1*torch.sin(0.01*z)+0.1*torch.sin(0.1*z)+0.1*torch.sin(z))
    Mat[:,1]=matMean[1]*(1+0.1*torch.sin(0.009*z)+0.1*torch.sin(0.09*z)+0.1*torch.sin(z))
    Mat[:,2]=matMean[2]*(1+0.1*torch.sin(0.008*z)+0.1*torch.sin(0.08*z)+0.1*torch.sin(z))
    Mat[:,3]=matMean[3]*(1+0.1*torch.sin(0.007*z)+0.1*torch.sin(0.07*z)+0.1*torch.sin(z))
    Mat[:,4]=matMean[4] # mean orientation
    Mat[:,5]=matMean[5] # 1e5, a known constant
    return Mat

def generate_mat_distribution3(folder='./data/'):
    mesh_tube=PolyhedronMesh()
    mesh_tube.load_from_torch(folder+'aorta_tube_solid_1layers.pt')
    node=mesh_tube.node
    element=mesh_tube.element
    matMean=torch.load(folder+'125mat.pt')['mean_mat']
    #element center
    center=node[element].mean(dim=1)
    z=center[:,2]*100
    Mat=torch.zeros((element.shape[0], 6), dtype=torch.float64)
    Mat[:,0]=matMean[0]*(1+0.1*torch.sin(0.01*z)+0.1*torch.sin(0.1*z)+0.1*torch.sin(z))
    Mat[:,1]=matMean[1]*(1+0.1*torch.sin(0.009*z)+0.1*torch.sin(0.09*z)+0.1*torch.sin(z))
    Mat[:,2]=matMean[2]*(1+0.1*torch.sin(0.008*z)+0.1*torch.sin(0.08*z)+0.1*torch.sin(z))
    Mat[:,3]=matMean[3]*(1+0.1*torch.sin(0.007*z)+0.1*torch.sin(0.07*z)+0.1*torch.sin(z))
    Mat[:,4]=matMean[4] # mean orientation
    Mat[:,5]=matMean[5] # 1e5, a known constant
    return Mat
