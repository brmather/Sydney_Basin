#!/usr/bin/env python
# coding: utf-8

# # 2 - Initialise Underworld model
# 
# Set up the Underworld model with the appropriate material properties to solve steady state Darcy flow.

# In[1]:


import numpy as np
import os
import stripy
from scipy import interpolate
from scipy.spatial import cKDTree

import underworld as uw


# In[2]:


data_dir = "../Data/"

with np.load(data_dir+"sydney_basin_surfaces.npz", "r") as npz:
    grid_list    = npz["layers"]
    grid_Xcoords = npz['Xcoords']
    grid_Ycoords = npz['Ycoords']
    
xmin, xmax = float(grid_Xcoords.min()), float(grid_Xcoords.max())
ymin, ymax = float(grid_Ycoords.min()), float(grid_Ycoords.max())
zmin, zmax = float(grid_list.min()),    float(grid_list.max())

print("x {:8.3f} -> {:8.3f} km".format(xmin/1e3,xmax/1e3))
print("y {:8.3f} -> {:8.3f} km".format(ymin/1e3,ymax/1e3))
print("z {:8.3f} -> {:8.3f} km".format(zmin/1e3,zmax/1e3))


# ## Set up the mesh

# In[3]:


## setup model resolution

# global size
Nx, Ny, Nz = 20, 20, 50


deformedmesh = True
elementType = "Q1"
mesh = uw.mesh.FeMesh_Cartesian( elementType = (elementType), 
                                 elementRes  = (Nx,Ny,Nz), 
                                 minCoord    = (xmin,ymin,zmin), 
                                 maxCoord    = (xmax,ymax,zmax)) 

gwPressureField            = mesh.add_variable( nodeDofCount=1 )
velocityField              = mesh.add_variable( nodeDofCount=3 )


# In[8]:


coords = mesh.data

Xcoords = np.unique(coords[:,0])
Ycoords = np.unique(coords[:,1])
Zcoords = np.unique(coords[:,2])
nx, ny, nz = Xcoords.size, Ycoords.size, Zcoords.size


# ### Import geological surfaces
# 
# Load numpy archive with the surfaces we will use.

# In[9]:


with np.load(data_dir+"sydney_basin_surfaces.npz", "r") as npz:
    grid_list    = npz["layers"]
    grid_Xcoords = npz['Xcoords']
    grid_Ycoords = npz['Ycoords']

# set up interpolation object
interp = interpolate.RegularGridInterpolator((grid_Ycoords, grid_Xcoords), grid_list[0], method="nearest")

# update grid list for top and bottom of model
grid_list = list(grid_list)
grid_list.append(np.full_like(grid_list[0], zmin))
grid_list = np.array(grid_list)

n_layers = grid_list.shape[0]


# ### Wrap mesh to surface topography
# 
# We want to deform z component so that we bunch up the mesh where we have valleys. The total number of cells should remain the same, only the vertical spacing should vary.

# In[19]:


interp.values = grid_list[0]
local_topography = interp((mesh.data[:,1],mesh.data[:,0]))

# depth above which to deform
z_deform = zmin

with mesh.deform_mesh():
    zcube = coords[:,2].reshape(nz,ny,nx)
    zcube_norm = zcube.copy()
    zcube_norm -= z_deform
    zcube_norm /= zmax - z_deform
    zcube_mask = zcube_norm < 0
    
    # difference to add to existing z coordinates
    dzcube = zcube_norm * -(zmax - local_topography.reshape(zcube.shape))
    
    mesh.data[:,2] += dzcube.ravel()
    coords = mesh.data


# ## Set up the types of boundary conditions
# 
# We'll set the left, right and bottom walls such that flow cannot pass through them, only parallel.
# In other words for groundwater pressure $P$:
# 
# $ \frac{\partial P}{\partial x}=0$ : left and right walls
# 
# $ \frac{\partial P}{\partial y}=0$ : bottom wall
# 
# This is only solvable if there is topography or a non-uniform upper pressure BC.

# In[20]:


topWall = mesh.specialSets["MaxK_VertexSet"]
bottomWall = mesh.specialSets["MinK_VertexSet"]

gwPressureBC = uw.conditions.DirichletCondition( variable        = gwPressureField, 
                                               indexSetsPerDof = ( topWall   ) )


# In[21]:


# lower groundwater pressure BC - value is relative to gravity
maxgwpressure = 0.9

znorm = mesh.data[:,2].copy()
znorm -= zmin
znorm /= zmax
linear_gradient = 1.0 - znorm

initial_pressure = linear_gradient*maxgwpressure
gwPressureField.data[:] = initial_pressure.reshape(-1,1)


# ## Set up the swarm particles
# 
# It is best to set only one particle per cell, to prevent variations in hydaulic diffusivity within cells.

# In[26]:


swarm         = uw.swarm.Swarm( mesh=mesh )
swarmLayout   = uw.swarm.layouts.PerCellGaussLayout(swarm=swarm, gaussPointCount=4)
swarm.populate_using_layout( layout=swarmLayout )


# __Assign materials to particles.__

# In[27]:


materialIndex        = swarm.add_variable( dataType="int",    count=1 )
swarmVelocity        = swarm.add_variable( dataType="double", count=3 )
hydraulicDiffusivity = swarm.add_variable( dataType="double", count=1 )


# In[28]:


for cell in range(0, mesh.elementsLocal):
    mask_cell = swarm.owningCell.data == cell
    idx_cell  = np.nonzero(mask_cell)[0]
    cell_centroid = swarm.data[idx_cell].mean(axis=0)
    cx, cy, cz = cell_centroid

    for lith in range(n_layers):
        interp.values = grid_list[lith]
        z_interp = interp((cy,cx))

        # starting from surface and going deeper with each layer
        if cz > z_interp:
            break
            
    materialIndex.data[mask_cell] = lith

print(uw.mpi.rank, materialIndex.data.min(), materialIndex.data.max(), n_layers)
print(uw.mpi.rank, np.bincount(materialIndex.data.ravel(), minlength=n_layers))
# ### Assign material properties
# 
# Use level sets to assign hydraulic diffusivities to a region on the mesh corresponding to any given material index.
# 
# - $H$ : rate of heat production
# - $\rho$ : density
# - $k_h$ : hydraulic conductivity
# - $k_t$ : thermal conductivity
# - $\kappa_h$ : hydraulic diffusivity
# - $\kappa_t$ : thermal diffusivity
# 
# __First, there are some lithologies that need to be collapsed.__

# In[29]:


def read_material_index(filename, cols):
    """
    Reads the material index with specified columns
    """
    import csv
    
    layerIndex = dict()
    
    with open(filename, 'r') as f:
        rowreader = csv.reader(f, delimiter=',')
        csvdata = list(rowreader)
        header = csvdata.pop(0)
        
    nrows = len(csvdata)

    matName  = []
    matIndex = np.empty(nrows, dtype=int)
    read_columns = np.empty((nrows, len(cols)))
    
    for r, row in enumerate(csvdata):
        index = int(row[0])
        matIndex[r] = index
        layerIndex[index] = np.array(row[1].split(' '), dtype=int)
        matName.append(row[2])
        
        for c, col in enumerate(cols):
            read_columns[r,c] = row[col]
            
    return layerIndex, matIndex, matName, list(read_columns.T)


# In[30]:


cols = [3,5,7,9]
layerIndex, matIndex, matName, [rho, kt, H, kh] = read_material_index(data_dir+"material_properties.csv", cols)


# In[31]:


voxel_model_condensed = materialIndex.data.flatten()

# condense lith(s) to index(s)
for index in matIndex:
    for lith in layerIndex[index]:
        voxel_model_condensed[voxel_model_condensed == lith] = index
        
# populate mesh variables with material properties
for i, index in enumerate(matIndex):
    mask_material = voxel_model_condensed == index
    hydraulicDiffusivity.data[mask_material] = kh[i]


# ## Set up groundwater equations

# In[32]:


if deformedmesh:
    g = uw.function.misc.constant((0.,0.,-1.))
else:
    g = uw.function.misc.constant((0.,0.,0.))
    
Storage = 1.
rho_water = 1.

gwPressureGrad = gwPressureField.fn_gradient

gMapFn = -g*rho_water*Storage
gwadvDiff = uw.systems.SteadyStateDarcyFlow(
                                            velocityField    = velocityField, \
                                            pressureField    = gwPressureField, \
                                            fn_diffusivity   = hydraulicDiffusivity, \
                                            conditions       = [gwPressureBC], \
                                            fn_bodyforce     = -gMapFn, \
                                            voronoi_swarm    = swarm, \
                                            swarmVarVelocity = swarmVelocity)

gwsolver = uw.systems.Solver(gwadvDiff)


# In[33]:


gwsolver.solve()


# __Save to HDF5__

# In[34]:


# project hydraulic diffusivity to the mesh
hydraulicDiffusivityField = mesh.add_variable( nodeDofCount=1 )
materialIndexField        = mesh.add_variable( nodeDofCount=1 )

hydproj = uw.utils.MeshVariable_Projection(hydraulicDiffusivityField, hydraulicDiffusivity, swarm)
hydproj.solve()

hydproj = uw.utils.MeshVariable_Projection(materialIndexField, materialIndex, swarm)
hydproj.solve()


# In[35]:



xdmf_info_mesh  = mesh.save('mesh.h5')
xdmf_info_swarm = swarm.save('swarm.h5')

xdmf_info_matIndex = materialIndex.save('materialIndex.h5')
materialIndex.xdmf('materialIndex.xdmf', xdmf_info_matIndex, 'materialIndex', xdmf_info_swarm, 'TheSwarm')


# In[36]:


for xdmf_info,save_name,save_object in [(xdmf_info_mesh,  'hydraulicDiffusivityField', hydraulicDiffusivityField),
                                        (xdmf_info_mesh,  'velocityField', velocityField),
                                        (xdmf_info_mesh,  'pressureField', gwPressureField),
                                        (xdmf_info_mesh,  'materialIndexField', materialIndexField),
                                        (xdmf_info_swarm, 'hydraulicDiffusivitySwarm', hydraulicDiffusivity)]:
    
    xdmf_info_var = save_object.save(save_name+'.h5')
    save_object.xdmf(save_name+'.xdmf', xdmf_info_var, save_name, xdmf_info, 'TheMesh')


# In[ ]:




