# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 22:54:53 2022

@author: liang
"""

import torch
#%%
def save_polygon_mesh_to_vtk(mesh, filename):
    out=[]
    out.append('# vtk DataFile Version 4.2'+'\n')
    out.append('vtk output'+'\n')
    out.append('ASCII'+'\n')
    out.append('DATASET POLYDATA'+'\n')
    #------------------------------------------------------------------
    out.append('POINTS '+str(mesh.node.shape[0])+' double'+'\n')
    node=mesh.node
    if isinstance(node, torch.Tensor):
        node=node.detach().to('cpu')
    for n in range(0, mesh.node.shape[0]):
        x=float(node[n,0])
        y=float(node[n,1])
        z=float(node[n,2])
        out.append(str(x)+' '+str(y)+' '+str(z)+'\n')
    #------------------------------------------------------------------
    element=mesh.element
    if isinstance(element, torch.Tensor):
        element=element.detach().to('cpu')
    offset_count=0
    for m in range(0, len(element)):
        offset_count+=1+len(element[m])
    out.append('POLYGONS '+str(len(element))+' '+str(offset_count)+'\n')
    for m in range(0, len(element)):
        line=str(len(element[m]))
        for k in range(0, len(element[m])):
            id=int(element[m][k])
            line=line+' '+str(id)
        out.append(line+'\n')
    #------------------------------------------------------------------
    if len(mesh.node_data.keys()) > 0:
        out.append('POINT_DATA '+str(node.shape[0])+'\n')
        out.append('FIELD FieldData '+str(len(mesh.node_data.keys())))
    for name, data in mesh.node_data.items():
        out.append(name+' '+str(data.shape[1])+' '+str(data.shape[0])+' double'+'\n')
        if isinstance(data, torch.Tensor):
            data=data.detach().to('cpu')
        for i in range(0, data.shape[0]):
            line=''
            for j in range(data.shape[1]):
                line=line+str(float(data[i,j]))+' '
            out.append(line+'\n')
    #------------------------------------------------------------------
    if len(mesh.element_data.keys()) > 0:
        out.append('CELL_DATA '+str(len(element))+'\n')
        out.append('FIELD FieldData '+str(len(mesh.element_data.keys())))
    for name, data in mesh.element_data.items():
        out.append(name+' '+str(data.shape[1])+' '+str(data.shape[0])+' double'+'\n')
        if isinstance(data, torch.Tensor):
            data=data.detach().to('cpu')
        for i in range(0, data.shape[0]):
            line=''
            for j in range(data.shape[1]):
                line=line+str(float(data[i,j]))+' '
            out.append(line+'\n')
    #------------------------------------------------------------------
    with open(filename, 'w', encoding = 'utf-8') as file:
        file.writelines(out)

#%%
def save_polyhedron_mesh_to_vtk(mesh, filename):
    out=[]
    out.append('# vtk DataFile Version 4.2'+'\n')
    out.append('vtk output'+'\n')
    out.append('ASCII'+'\n')
    out.append('DATASET UNSTRUCTURED_GRID'+'\n')
    #------------------------------------------------------------------
    out.append('POINTS '+str(mesh.node.shape[0])+' double'+'\n')
    node=mesh.node
    if isinstance(node, torch.Tensor):
        node=node.detach().to('cpu')
    for n in range(0, mesh.node.shape[0]):
        x=float(node[n,0])
        y=float(node[n,1])
        z=float(node[n,2])
        out.append(str(x)+' '+str(y)+' '+str(z)+'\n')
    #------------------------------------------------------------------
    element=mesh.element
    if isinstance(element, torch.Tensor):
        element=element.detach().to('cpu')
    offset_count=0
    for m in range(0, len(element)):
        offset_count+=1+len(element[m])
    out.append('CELLS '+str(len(element))+' '+str(offset_count)+'\n')
    for m in range(0, len(element)):
        line=str(len(element[m]))
        for k in range(0, len(element[m])):
            id=int(element[m][k])
            line=line+' '+str(id)
        out.append(line+'\n')
    #------------------------------------------------------------------
    out.append('CELL_TYPES '+str(len(element))+'\n')
    for m in range(0, len(element)):
        n_nodes=len(element[m])
        if n_nodes == 4:
            #cell_type=vtk.VTK_TETRA
            out.append('10'+'\n')
        elif n_nodes == 6:
            #cell_type=vtk.VTK_WEDGE
            out.append('13'+'\n')
        elif n_nodes == 8:
            #cell_type=vtk.VTK_HEXAHEDRON
            out.append('12'+'\n')
        elif n_nodes == 10:
            out.append('24'+'\n')
            #cell_type=vtk.TK_QUADRATIC_TETRA
        else:
            #cell_type=vtk.VTK_POLYHEDRON
            out.append('42'+'\n')
    #------------------------------------------------------------------
    if len(mesh.node_data.keys()) > 0:
        out.append('POINT_DATA '+str(node.shape[0])+'\n')
        out.append('FIELD FieldData '+str(len(mesh.node_data.keys())))
    for name, data in mesh.node_data.items():
        out.append(name+' '+str(data.shape[1])+' '+str(data.shape[0])+' double'+'\n')
        if isinstance(data, torch.Tensor):
            data=data.detach().to('cpu')
        for i in range(0, data.shape[0]):
            line=''
            for j in range(data.shape[1]):
                line=line+str(float(data[i,j]))+' '
            out.append(line+'\n')
    #------------------------------------------------------------------
    if len(mesh.element_data.keys()) > 0:
        out.append('CELL_DATA '+str(len(element))+'\n')
        out.append('FIELD FieldData '+str(len(mesh.element_data.keys())))
    for name, data in mesh.element_data.items():
        out.append(name+' '+str(data.shape[1])+' '+str(data.shape[0])+' double'+'\n')
        if isinstance(data, torch.Tensor):
            data=data.detach().to('cpu')
        for i in range(0, data.shape[0]):
            line=''
            for j in range(data.shape[1]):
                line=line+str(float(data[i,j]))+' '
            out.append(line+'\n')
    #------------------------------------------------------------------
    with open(filename, 'w', encoding = 'utf-8') as file:
        file.writelines(out)