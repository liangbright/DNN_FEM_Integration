# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 19:05:36 2021

@author: liang
"""
import torch
from Element_C3D8 import cal_strain_energy
from FEModel_C3D8 import cal_potential_energy
from FEModel_C3D8 import cal_F_tensor_1i, cal_F_tensor_8i
from FEModel_C3D8 import cal_d_sf_dX_and_dX_dr_1i, cal_d_sf_dX_and_dX_dr_8i
from FEModel_C3D8 import cal_d_sf_dx_and_dx_dr_1i, cal_d_sf_dx_and_dx_dr_8i
from FEModel_C3D8 import cal_nodal_force_from_1pk_stress_8i, cal_nodal_force_from_1pk_stress_1i
from FEModel_C3D8 import cal_nodal_force_from_cauchy_stress_8i, cal_nodal_force_from_cauchy_stress_1i
from FEModel_C3D8 import cal_pressure_force_quad_4i, cal_pressure_force_quad_1i
from FEModel_C3D8 import cal_attribute_on_node
from FEModel_C3D8 import cal_cauchy_stress_force as _cal_cauchy_stress_force_
from FEModel_C3D8 import cal_1pk_stress_force as _cal_1pk_stress_force_
from FEModel_C3D8 import cal_cauchy_stress_force_inverse as _cal_cauchy_stress_force_inverse_
#%%
def cal_cauchy_stress_force(node_x, element, d_sf_dX, material, element_orientation, cal_cauchy_stress,
                            F0=None, return_F_S_W=False, return_stiffness=None, return_force_of_element=False):
    def _cal_cauchy_stress_(F, material, create_graph, return_W):
        return cal_cauchy_stress(F, material, element_orientation, create_graph, return_W)
    return _cal_cauchy_stress_force_(node_x, element, d_sf_dX, material, _cal_cauchy_stress_,
                                     F0, return_F_S_W, return_stiffness, return_force_of_element)
#%%
def cal_cauchy_stress_force_inverse(d_sf_dx, det_dx_dr, n_nodes, element,
                                    F, material, element_orientation, cal_cauchy_stress,
                                    F0=None, return_S_W=False):
    def _cal_cauchy_stress_(F, material, create_graph, return_W):
        return cal_cauchy_stress(F, material, element_orientation, create_graph, return_W)
    return _cal_cauchy_stress_force_inverse_(d_sf_dx, det_dx_dr, n_nodes, element,
                                             F, material, _cal_cauchy_stress_,
                                             F0, return_S_W)
#%%
def cal_1pk_stress_force(node_x, element, d_sf_dX, det_dX_dr, material, element_orientation, cal_1pk_stress,
                         F0=None, return_F_S_W=False, return_stiffness=None, return_force_of_element=False):
    def _cal_1pk_stress_(F, material, create_graph, return_W):
        return cal_1pk_stress(F, material, element_orientation, create_graph, return_W)
    return _cal_1pk_stress_force_(node_x, element, d_sf_dX, det_dX_dr, material, _cal_1pk_stress_,
                                  F0, return_F_S_W, return_stiffness, return_force_of_element)
