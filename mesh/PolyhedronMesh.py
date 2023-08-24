# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
import numpy as np
from Mesh import Mesh
#%%
class PolyhedronMesh(Mesh):
    # use this class to handle mixture of tetra and hex elements
    def __init__(self):
        super().__init__('polyhedron')

#%%
if __name__ == "__main__":
    #
    root=PolyhedronMesh()
    root.load_from_vtk("D:/MLFEA/TAA/data/343c1.5/matMean/p0_0_solid_matMean_p20.vtk", 'float64')
    root.save_by_vtk("D:/MLFEA/TAA/test_PolyhedronMesh.vtk")
