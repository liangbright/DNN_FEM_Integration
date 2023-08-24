# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 19:05:36 2021

@author: liang
"""
import torch
from FEModel_C3D8 import cal_potential_energy
from FEModel_C3D8 import cal_F_tensor_8i, cal_F_tensor_1i, cal_F_tensor_with_d_sf_dX
from FEModel_C3D8 import cal_d_sf_dx_and_dx_dr_8i, cal_d_sf_dx_and_dx_dr_1i
from FEModel_C3D8 import cal_d_sf_dX_and_dX_dr_8i, cal_d_sf_dX_and_dX_dr_1i
from FEModel_C3D8 import cal_nodal_force_from_1pk_stress_8i, cal_nodal_force_from_1pk_stress_1i
from FEModel_C3D8 import cal_nodal_force_from_cauchy_stress_8i, cal_nodal_force_from_cauchy_stress_1i
from FEModel_C3D8 import cal_pressure_force_quad_4i, cal_pressure_force_quad_1i
from FEModel_C3D8 import cal_attribute_on_node
from FEModel_C3D8_SRI import cal_cauchy_stress_force as _cal_cauchy_stress_force_
from FEModel_C3D8_SRI import cal_1pk_stress_force as _cal_1pk_stress_force_
from FEModel_C3D8_SRI import cal_cauchy_stress_force_inverse as _cal_cauchy_stress_force_inverse_
from FEModel_C3D8_SRI import cal_strain_energy
#%%
def cal_cauchy_stress_force(node_x, element, d_sf_dX_8i, d_sf_dX_1i, material, element_orientation, cal_cauchy_stress,
                            F0_8i=None, F0_1i=None,
                            return_F_S_W=False, return_stiffness=None, return_force_of_element=False):
    def _cal_cauchy_stress_(F_8i, F_1i, material, create_graph, return_W):
        return cal_cauchy_stress(F_8i, F_1i, material, element_orientation, create_graph, return_W)
    return _cal_cauchy_stress_force_(node_x, element, d_sf_dX_8i, d_sf_dX_1i, material, _cal_cauchy_stress_,
                                     F0_8i, F0_1i, return_F_S_W, return_stiffness, return_force_of_element)
#%%
def cal_cauchy_stress_force_inverse(d_sf_dx_8i, d_sf_dx_1i, det_dx_dr_8i, det_dx_dr_1i,
                                    n_nodes, element, F_8i, F_1i, material, element_orientation, cal_cauchy_stress,
                                    F0_8i=None, F0_1i=None, return_S_W=False):
    def _cal_cauchy_stress_(F_8i, F_1i, material, create_graph, return_W):
        return cal_cauchy_stress(F_8i, F_1i, material, element_orientation, create_graph, return_W)
    return _cal_cauchy_stress_force_inverse_(d_sf_dx_8i, d_sf_dx_1i, det_dx_dr_8i, det_dx_dr_1i,
                                             n_nodes, element, F_8i, F_1i, material, _cal_cauchy_stress_,
                                             F0_8i, F0_1i, return_S_W)
#%%
def cal_1pk_stress_force(node_x, element, d_sf_dX_8i, d_sf_dX_1i, det_dX_dr_8i, det_dX_dr_1i,
                         material, element_orientation, cal_1pk_stress, F0_8i=None, F0_1i=None,
                         return_F_S_W=False, return_stiffness=None, return_force_of_element=False):
    def _cal_1pk_stress_(F_8i, F_1i, material, create_graph, return_W):
        return cal_1pk_stress(F_8i, F_1i, material, element_orientation, create_graph, return_W)
    return _cal_1pk_stress_force_(node_x, element, d_sf_dX_8i, d_sf_dX_1i, det_dX_dr_8i, det_dX_dr_1i,
                                  material, _cal_1pk_stress_, F0_8i, F0_1i,
                                  return_F_S_W, return_stiffness, return_force_of_element)
