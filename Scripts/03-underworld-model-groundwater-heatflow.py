#!/usr/bin/env python
# coding: utf-8

# # 3 - Underworld model: groundwater + heat flow
# 
# Set up the Underworld model with the appropriate material properties to solve steady state Darcy flow and heat flow.

import numpy as np
import os
import argparse
import stripy
from scipy import interpolate
from scipy.spatial import cKDTree

import underworld as uw


parser = argparse.ArgumentParser(description='Process some model arguments.')
parser.add_argument('echo', type=str, metavar='PATH', help='I/O location')
parser.add_argument('-v', action='store_true', required=False, default=False, help="Verbose output")
args = parser.parse_args()


data_dir = args.echo
verbose  = args.v
Tmin = 0.0
Tmax = 100.0

xmin, xmax = 1e99, 1e-99
ymin, ymax = 1e99, 1e-99
zmin, zmax = 1e99, 1e-99

csvfiles = []
for i, f in enumerate(sorted(os.listdir(data_dir))):
    if f.endswith('.csv') and f[0:2].isdigit():
        csvfiles.append(f)
        
        xyz = np.loadtxt(data_dir+f, delimiter=',', skiprows=1)
        
        xmin, ymin, zmin = np.minimum([xmin, ymin, zmin], xyz.min(axis=0))
        xmax, ymax, zmax = np.maximum([xmax, ymax, zmax], xyz.max(axis=0))
        
        if uw.mpi.rank == 0 and verbose:
            print("{:35} : av depth {:10.3f} +- {:9.3f} metres".format(f, xyz[:,-1].mean(), np.std(xyz[:,-1])))

csvfiles = list(sorted(csvfiles))

if uw.mpi.rank ==0 and verbose:
    print(" ")
    print("x {:8.3f} -> {:8.3f} km".format(xmin/1e3,xmax/1e3))
    print("y {:8.3f} -> {:8.3f} km".format(ymin/1e3,ymax/1e3))
    print("z {:8.3f} -> {:8.3f} km".format(zmin/1e3,zmax/1e3))


## Set up the mesh

# global size
Nx, Ny, Nz = 20, 20, 50


deformedmesh = True
elementType = "Q1"
mesh = uw.mesh.FeMesh_Cartesian( elementType = (elementType), 
                                 elementRes  = (Nx,Ny,Nz), 
                                 minCoord    = (xmin,ymin,zmin), 
                                 maxCoord    = (xmax,ymax,zmax)) 

gwPressureField            = mesh.add_variable( nodeDofCount=1 )
temperatureField           = mesh.add_variable( nodeDofCount=1 )
velocityField              = mesh.add_variable( nodeDofCount=3 )


Xcoords = np.unique(mesh.data[:,0])
Ycoords = np.unique(mesh.data[:,1])
Zcoords = np.unique(mesh.data[:,2])
nx, ny, nz = Xcoords.size, Ycoords.size, Zcoords.size

tree = cKDTree(mesh.data)

xq, yq = np.meshgrid(Xcoords, Ycoords)
xq_ = xq.ravel()
yq_ = yq.ravel()
tree_layer = cKDTree(np.c_[xq_,yq_])

voxel_model = np.full((nz,ny,nx), -1, dtype=np.int)
layer_mask = np.zeros(nz*ny*nx, dtype=bool)



z_grid_prev = np.zeros((ny,nx))

grid_list = []

for lith, f in enumerate(csvfiles):

    # load the surface and remove duplicates
    x,y,z = np.loadtxt(data_dir+f, delimiter=',', skiprows=1, unpack=True)
    unique_xy, unique_index = np.unique(np.c_[x,y], axis=0, return_index=True)
    x, y = list(unique_xy.T)
    z = z[unique_index]

    # interpolate to grid
    smesh = stripy.Triangulation(x, y, tree=True, permute=True)
    smesh.update_tension_factors(z)
    z_grid = smesh.interpolate_to_grid(Xcoords, Ycoords, z)
    z_near, zierr = smesh.interpolate_nearest(xq_, yq_, z)
    z_near = z_near.reshape(z_grid.shape)

    # set entries outside tolerance to nearest neighbour interpolant
    ztol = 0.1*(np.percentile(z,99) - np.percentile(z,1))
    z_mask = np.logical_or(z_grid > z_near + ztol, z_grid < z_near - ztol)
    z_grid[z_mask] = z_near[z_mask]

    # avoid the gap - make sure we interpolate where we have data
    d, idx = smesh.nearest_vertices(xq_, yq_)
    dtol = np.sqrt(smesh.areas()).mean()
    z_inv_mask = np.logical_or(zierr==1, d > dtol)
    z_grid.flat[z_inv_mask] = z_grid_prev.flat[z_inv_mask]

    # interpolate to voxel_model
    d, idx = tree.query(np.c_[xq_, yq_, z_grid.ravel()])
    layer_mask.fill(0)
    layer_mask[idx] = True
    i0, j0, k0 = np.where(layer_mask.reshape(nz,ny,nx))
    for i in range(i0.size):
        voxel_model[:i0[i], j0[i], k0[i]] = lith+1


    # store for next surface
    z_grid_prev = z_grid.copy()
    
    grid_list.append(z_grid.copy())


# ### Wrap mesh to surface topography
# 
# We want to deform z component so that we bunch up the mesh where we have valleys.
# The total number of cells should remain the same, only the vertical spacing should vary.


local_topography = grid_list[0]

# depth above which to deform
z_deform = zmin

with mesh.deform_mesh():
    zcube = mesh.data[:,2].reshape(nz,ny,nx)
    zcube_norm = zcube.copy()
    zcube_norm -= z_deform
    zcube_norm /= zmax - z_deform
    zcube_mask = zcube_norm < 0
    
    # difference to add to existing z coordinates
    dzcube = zcube_norm * -(zmax - local_topography)
    
    mesh.data[:,2] += dzcube.ravel()


# ## Set up the types of boundary conditions
# 
# We'll set the left, right and bottom walls such that flow cannot pass through them, only parallel.
#
# This is only solvable if there is topography or a non-uniform upper pressure BC.


topWall = mesh.specialSets["MaxK_VertexSet"]
bottomWall = mesh.specialSets["MinK_VertexSet"]

gwPressureBC = uw.conditions.DirichletCondition( variable      = gwPressureField, 
                                               indexSetsPerDof = ( topWall   ) )

temperatureBC = uw.conditions.DirichletCondition( variable        = temperatureField,
                                                  indexSetsPerDof = (topWall+bottomWall))





# lower groundwater pressure BC - value is relative to gravity
maxgwpressure = 0.9

# zCoordFn = uw.function.input()[2]
# yCoordFn = uw.function.input()[1]
# xCoordFn = uw.function.input()[0]


# upper groundwater pressure set to topography
# if deformedMesh then the initial pressure field is just a smooth gradient
linear_gradient = 1.0 - zcube_norm.ravel()

initial_pressure = linear_gradient*maxgwpressure
initial_temperature = linear_gradient*(Tmax - Tmin) + Tmin


gwPressureField.data[:]  = initial_pressure.reshape(-1,1)
temperatureField.data[:] = initial_temperature.reshape(-1,1)


# ## Set up the swarm particles
# 
# It is best to set only one particle per cell, to prevent variations in hydaulic diffusivity within cells.

gaussPointCount = 4

swarm         = uw.swarm.Swarm( mesh=mesh )
swarmLayout   = uw.swarm.layouts.PerCellGaussLayout(swarm=swarm,gaussPointCount=gaussPointCount)
swarm.populate_using_layout( layout=swarmLayout )


# *Assign materials to particles.*
# 
# This requires interpolating from mesh to swarm. You would think this should be quick (but it ain't!):
# 
# ```python
# # set up swarm variable & mesh variable
# materialIndex = swarm.add_variable( dataType='int', count=1 )
# materialIndex_mesh = mesh.add_variable( nodeDofCount=1 )
# materialIndex_mesh.data[:] = voxel_model.reshape(-1,1)
# 
# # evaluate mesh variable on the swarm
# materialIndex.data[:] = materialIndex_mesh.evaluate(swarm)
# ```
# 
# Instead, I make do with __k-d trees__!

# In[ ]:


materialIndex  = swarm.add_variable( dataType="int",    count=1 )
swarmVelocity  = swarm.add_variable( dataType="double", count=3 )

hydraulicDiffusivity = swarm.add_variable( dataType="double", count=1 )
thermalDiffusivity   = swarm.add_variable( dataType="double", count=1 )
heatProduction       = swarm.add_variable( dataType="double", count=1 )


# In[ ]:


# mesh has been deformed, rebuild k-d tree
tree_mesh = cKDTree(mesh.data)
d, idx = tree_mesh.query(swarm.data)

# project from mesh to the swarm
materialIndex.data[:] = voxel_model.flat[idx].reshape(-1,1)


# ### Assign material properties
# 
# Use level sets to assign hydraulic diffusivities to a region on the mesh corresponding to any given material index.
# 
# - H       : rate of heat production
# - rho     : density
# - k_h     : hydraulic conductivity
# - k_t     : thermal conductivity
# - kappa_h : hydraulic diffusivity
# - kappa_t : thermal diffusivity
# 
# First, there are some lithologies that need to be collapsed.

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



cols = [3,5,7,9]
layerIndex, matIndex, matName, [rho, kt, H, kh] = read_material_index(data_dir+"material_properties.csv", cols)



voxel_model_condensed = voxel_model.flatten()
voxel_model_swarm     = materialIndex.data.copy()

# condense lith(s) to index(s)
for index in matIndex:
    for lith in layerIndex[index]:
        voxel_model_condensed[voxel_model_condensed == lith] = index
        voxel_model_swarm[voxel_model_swarm == lith] = index
        
# populate mesh variables with material properties
for i, index in enumerate(matIndex):
    mask_material = voxel_model_swarm == index
    hydraulicDiffusivity.data[mask_material] = kh[i]
    thermalDiffusivity.data[mask_material]   = kt[i]
    heatProduction.data[mask_material]       = H[i]


# **Ensure `materialIndex` is constant within an element!**
# 
# Take the min/max of all cell centroids. If they are different then the swarm values
# assigned within that cell should be averaged.


# be careful that elements in data_elementNodes are not global IDs
elementNodes = mesh.data_elementNodes - mesh.data_elementNodes.min()
assert elementNodes.shape[0] == mesh.elementsLocal, "mismatch in number of elements"

element_centroids = mesh.data[elementNodes].mean(axis=1)
element_material_min = voxel_model.flat[elementNodes].min(axis=1)
element_material_max = voxel_model.flat[elementNodes].max(axis=1)

# find elements where max does not equal min
idx_nonconstant_cells = np.nonzero(element_material_min != element_material_max)[0]


owningCell = swarm.owningCell

for cell in idx_nonconstant_cells:
    mask_cell = swarm.owningCell.data == cell
    
    # reassign material properties
    hydraulicDiffusivity.data[mask_cell] = hydraulicDiffusivity.data[mask_cell].mean()
    thermalDiffusivity.data[mask_cell] = thermalDiffusivity.data[mask_cell].mean()
    heatProduction.data[mask_cell] = heatProduction.data[mask_cell].mean()


## Set up heat equation
if uw.mpi.rank == 0 and verbose:
    print("Solving heat equation...")

heateqn = uw.systems.SteadyStateHeat( temperatureField = temperatureField,
                                      fn_diffusivity   = thermalDiffusivity,
                                      fn_heating       = heatProduction,
                                      conditions       = temperatureBC
                                      )

heatsolver = uw.systems.Solver(heateqn)
heatsolver.solve()



if deformedmesh:
    g = uw.function.misc.constant((0.,0.,-1.))
else:
    g = uw.function.misc.constant((0.,0.,0.))


## Set up groundwater equation
if uw.mpi.rank == 0 and verbose:
    print("Solving grounwater equation...")

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
gwsolver.solve()



## Save to HDF5

xdmf_info_mesh  = mesh.save('mesh.h5')
xdmf_info_swarm = swarm.save('swarm.h5')

xdmf_info_matIndex = materialIndex.save('materialIndex.h5')
materialIndex.xdmf('materialIndex.xdmf', xdmf_info_matIndex, 'materialIndex', xdmf_info_swarm, 'TheSwarm')


# In[ ]:


for xdmf_info,save_name,save_object in [(xdmf_info_swarm, 'hydraulicDiffusivitySwarm', hydraulicDiffusivity),
                                        (xdmf_info_mesh, 'velocityField', velocityField),
                                        (xdmf_info_mesh, 'pressureField', gwPressureField),
                                        (xdmf_info_mesh, 'temperatureField', temperatureField),
                                        (xdmf_info_swarm, 'thermalDiffusivitySwarm', thermalDiffusivity),
                                        (xdmf_info_swarm, 'heatProductionSwarm', heatProduction)]:
    
    xdmf_info_var = save_object.save(save_name+'.h5')
    save_object.xdmf(save_name+'.xdmf', xdmf_info_var, save_name, xdmf_info, 'TheMesh')


