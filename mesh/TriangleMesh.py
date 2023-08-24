# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
import torch_scatter
from torch_sparse import SparseTensor
import numpy as np
from PolygonMesh import PolygonMesh
#%%
class TriangleMesh(PolygonMesh):
    #3-node triangle element mesh
    def __init__(self):
        super().__init__()
        self.mesh_type='polygon_tri3'
        self.node_normal=None
        self.element_area=None
        self.element_normal=None

    def update_node_normal(self):
        self.node_normal=TriangleMesh.cal_node_normal(self.node, self.element)

    @staticmethod
    def cal_node_normal(node, element, element_normal=None, normalization=True):
        if element_normal is None:
            element_area, element_normal=TriangleMesh.cal_element_area_and_normal(node, element)
        M=element.shape[0]
        e_normal=element_normal.view(M, 1, 3)
        e_normal=e_normal.expand(M, 3, 3)
        e_normal=e_normal.reshape(M*3, 3)
        N=node.shape[0]
        normal = torch_scatter.scatter(e_normal, element.view(-1), dim=0, dim_size=N, reduce="sum")
        if normalization == True:
            normal_norm=torch.norm(normal, p=2, dim=1, keepdim=True)
            normal_norm=normal_norm.clamp(min=1e-12)
            normal=normal/normal_norm
        normal=normal.contiguous()
        return normal

    def update_element_area_and_normal(self):
         self.element_area, self.element_normal=TriangleMesh.cal_element_area_and_normal(self.node, self.element)

    @staticmethod
    def cal_element_area_and_normal(node, element):
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        #   x2
        #  /  \
        # x0--x1
        temp1=torch.cross(x1 - x0, x2- x0, dim=-1)
        temp2=torch.norm(temp1, p=2, dim=1, keepdim=True)
        area=0.5*temp2.abs()
        temp2=temp2.clamp(min=1e-12)
        normal=temp1/temp2
        return area, normal

    def sample_points_on_elements(self, n_points):
         return TriangleMesh.sample_points(self.node, self.element, n_points)

    @staticmethod
    def sample_points(node, element, n_points):
        area, normal=TriangleMesh.cal_element_area_and_normal(node, element)
        prob = area / area.sum()
        sample = torch.multinomial(prob.view(-1), n_points-len(element), replacement=True)
        #print("sample_points", area.shape, prob.shape, sample.shape)
        element = torch.cat([element, element[sample]], dim=0)
        a = torch.rand(2, n_points, 1, dtype=node.dtype, device=node.device)
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x=a[1]*(a[0]*x1+(1-a[0])*x2)+(1-a[1])*x0
        return x

    def subdivide(self):
        #return a new mesh
        #add a node in the middle of each edge
        if self.edge is None:
            self.build_edge()
        x_j=self.node[self.edge[:,0]]
        x_i=self.node[self.edge[:,1]]
        nodeA=(x_j+x_i)/2
        #create new mesh
        node_new=torch.cat([self.node, nodeA], dim=0)
        #adj matrix for nodeA
        adj=SparseTensor(row=self.edge[:,0],
                         col=self.edge[:,1],
                         value=torch.arange(self.node.shape[0],
                                            self.node.shape[0]+nodeA.shape[0]),
                         sparse_sizes=(nodeA.shape[0], nodeA.shape[0]))
        element_new=[]
        element=self.element.cpu().numpy()
        for m in range(0, element.shape[0]):
            #     x2
            #    /  \
            #   x5-- x4
            #  / \  / \
            # x0--x3--x1
            #-----------
            id0=element[m,0].item()
            id1=element[m,1].item()
            id2=element[m,2].item()
            if id0 < id1:
                id3=adj[id0, id1].to_dense().item()
            else:
                id3=adj[id1, id0].to_dense().item()
            if id1 < id2:
                id4=adj[id1, id2].to_dense().item()
            else:
                id4=adj[id2, id1].to_dense().item()
            if id2 < id0:
                id5=adj[id2, id0].to_dense().item()
            else:
                id5=adj[id0, id2].to_dense().item()
            element_new.append([id0, id3, id5])
            element_new.append([id3, id4, id5])
            element_new.append([id3, id1, id4])
            element_new.append([id5, id4, id2])
        element_new=torch.tensor(element_new, dtype=torch.int64, device=self.element.device)
        mesh_new=TriangleMesh()
        mesh_new.node=node_new
        mesh_new.element=element_new
        return mesh_new

    def get_sub_mesh(self, element_id_list):
        element_sub=self.element[element_id_list]
        node_idlist, element_out=torch.unique(element_sub.reshape(-1), return_inverse=True)
        node_new=self.node[node_idlist]
        element_new=element_out.view(-1,3)
        mesh_new=TriangleMesh()
        mesh_new.node=node_new
        mesh_new.element=element_new
        return mesh_new
#%%
if __name__ == "__main__":
    filename="wall_tri.vtk"
    wall=TriangleMesh()
    wall.load_from_vtk(filename, 'float64')
    wall.update_node_normal()
    wall.node+=wall.node_normal
    wall.save_by_vtk("wall_tri_offset.vtk")
    wall_sub = wall.subdivide()
    wall_sub.save_by_vtk("wall_tri_offset_sub.vtk")
    #%%
    wall.update_element_area_and_normal()
    #%%
    points=wall.sample_points_on_elements(10*len(wall.node))
    #%%
    wall_sub=wall.get_sub_mesh(torch.arange(0,100))
    wall_sub.save_by_vtk("wall_tri_sub.vtk")
