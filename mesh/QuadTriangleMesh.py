# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
import torch_scatter
from torch_sparse import SparseTensor
from PolygonMesh import PolygonMesh
from QuadMesh import QuadMesh
from TriangleMesh import TriangleMesh
from copy import deepcopy
#%%
class QuadTriangleMesh(PolygonMesh):
    #element could be quad or triangle

    def __init__(self):
        super().__init__()
        self.mesh_type='polygon_quad4_tri3'
        self.node_normal=None
        self.element_area=None
        self.element_normal=None
        self.quad_element=[]
        self.quad_element_idx=[]#index list of quad elements
        self.tri_element=[]
        self.tri_element_idx=[]#index list of tri elements

    def classify_element(self):
        if isinstance(self.element,  torch.Tensor):
            if len(self.element[0]) == 4:
                self.quad_element=self.element
                self.quad_element_idx=torch.arange(0, len(self.element))
            elif len(self.element[0]) == 3:
                self.tri_element=self.element
                self.tri_element_idx=torch.arange(0, len(self.element))
            else:
                raise ValueError("len(self.element[0])="+str(len(self.element[0])))
            return
        quad_element=[]
        quad_element_idx=[]
        tri_element=[]
        tri_element_idx=[]
        for m in range(0, len(self.element)):
            elm=self.element[m]
            if len(elm) == 4:
                quad_element.append(elm)
                quad_element_idx.append(m)
            elif len(elm) == 3:
                tri_element.append(elm)
                tri_element_idx.append(m)
            else:
                raise ValueError("len(elm)="+str(len(elm))+",m="+str(m))
        if len(quad_element_idx) > 0:
            self.quad_element=torch.tensor(quad_element, dtype=torch.int64)
            self.quad_element_idx=torch.tensor(quad_element_idx, dtype=torch.int64)
        if len(tri_element_idx) > 0:
            self.tri_element=torch.tensor(tri_element, dtype=torch.int64)
            self.tri_element_idx=torch.tensor(tri_element_idx, dtype=torch.int64)

    def load_from_vtk(self, filename, dtype):
        super().load_from_vtk(filename, dtype)
        self.classify_element()

    def load_from_torch(self, filename):
        super().load_from_torch(filename)
        self.classify_element()

    def quad_to_tri(self):
        super().quad_to_tri()
        self.classify_element()

    def copy(self, node, element, dtype=None, detach=True):
        super().copy(node, element, dtype, detach)
        self.classify_element()

    def update_node_normal(self):
        if len(self.quad_element) ==0 and len(self.tri_element) == 0:
            self.classify_element()
        normal_quad=0
        if len(self.quad_element) > 0:
            normal_quad=QuadMesh.cal_node_normal(self.node, self.quad_element, normalization=False)
        normal_tri=0
        if len(self.tri_element) > 0:
            normal_tri=TriangleMesh.cal_node_normal(self.node, self.tri_element, normalization=False)
        normal=normal_quad+normal_tri
        normal_norm=torch.norm(normal, p=2, dim=1, keepdim=True)
        normal_norm=normal_norm.clamp(min=1e-12)
        normal=normal/normal_norm
        normal=normal.contiguous()
        self.node_normal=normal

    def update_element_area_and_normal(self):
        if self.quad_element is None or self.tri_element is None:
            self.classify_element()
        area=torch.zeros((len(self.element) ,1), dtype=self.node.dtype, device=self.node.device)
        normal=torch.zeros((len(self.element), 3), dtype=self.node.dtype, device=self.node.device)
        if len(self.quad_element) > 0:
            area_quad, normal_quad=QuadMesh.cal_element_area_and_normal(self.node, self.quad_element)
            area[self.quad_element_idx]=area_quad
            normal[self.quad_element_idx]=normal_quad
        if len(self.tri_element) > 0:
            area_tri, normal_tri=TriangleMesh.cal_element_area_and_normal(self.node, self.tri_element)
            area[self.tri_element_idx]=area_tri
            normal[self.tri_element_idx]=normal_tri
        self.element_area=area
        self.element_normal=normal

    def subdivide(self):
        #return a new mesh
        #add a node in the middle of each quad element
        n_nodeA=0
        if len(self.quad_element) > 0:
            nodeA=self.node[self.quad_element].mean(dim=1) #(N,3) => (M,8,3) => (M,3)
            n_nodeA=nodeA.shape[0]
        #add a node in the middle of each edge
        if self.edge is None:
            self.build_edge()
        x_j=self.node[self.edge[:,0]]
        x_i=self.node[self.edge[:,1]]
        nodeB=(x_j+x_i)/2
        #create new mesh
        if n_nodeA > 0:
            node_new=torch.cat([self.node, nodeA, nodeB], dim=0)
        else:
            node_new=torch.cat([self.node, nodeB], dim=0)
        #adj matrix for nodeB
        adj=SparseTensor(row=self.edge[:,0],
                         col=self.edge[:,1],
                         value=torch.arange(self.node.shape[0]+n_nodeA,
                                            self.node.shape[0]+n_nodeA+nodeB.shape[0]),
                         sparse_sizes=(nodeB.shape[0], nodeB.shape[0]))
        element_new=[]
        for m in range(0, len(self.quad_element)):
            #-----------
            # x3--x6--x2
            # |   |   |
            # x7--x8--x5
            # |   |   |
            # x0--x4--x1
            #-----------
            elm=self.quad_element[m]
            id0=int(elm[0])
            id1=int(elm[1])
            id2=int(elm[2])
            id3=int(elm[3])
            if id0 < id1:
                id4=adj[id0, id1].to_dense().item()
            else:
                id4=adj[id1, id0].to_dense().item()
            if id1 < id2:
                id5=adj[id1, id2].to_dense().item()
            else:
                id5=adj[id2, id1].to_dense().item()
            if id2 < id3:
                id6=adj[id2, id3].to_dense().item()
            else:
                id6=adj[id3, id2].to_dense().item()
            if id3 < id0:
                id7=adj[id3, id0].to_dense().item()
            else:
                id7=adj[id0, id3].to_dense().item()
            id8=self.node.shape[0]+m
            element_new.append([id0, id4, id8, id7])
            element_new.append([id4, id1, id5, id8])
            element_new.append([id7, id8, id6, id3])
            element_new.append([id8, id5, id2, id6])
        for m in range(0, len(self.tri_element)):
            #-----------
            #     x2
            #    /  \
            #   x5-- x4
            #  / \  / \
            # x0--x3--x1
            #-----------
            elm=self.tri_element[m]
            id0=int(elm[0])
            id1=int(elm[1])
            id2=int(elm[2])
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
        mesh_new=QuadTriangleMesh()
        mesh_new.node=node_new
        mesh_new.element=element_new
        mesh_new.classify_element()
        return mesh_new

#%%
if __name__ == "__main__":
    filename="F:/MLFEA/TAA/data/343c1.5/bav17_AortaModel_P0_best.vtk"
    wall=QuadTriangleMesh()
    wall.load_from_vtk(filename, 'float64')
    wall.update_node_normal()
    wall.node+=1.5*wall.node_normal
    wall.save_by_vtk("aorta_quad_offset.vtk")


