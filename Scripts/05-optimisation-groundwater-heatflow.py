#!/usr/bin/env python
# coding: utf-8
# # 5 - Optimisation: groundwater + heat flow

# +
import numpy as np
import os
import argparse
from scipy import interpolate
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from mpi4py import MPI
comm = MPI.COMM_WORLD

import underworld as uw

# +
if uw.mpi.size > 1:
    parser = argparse.ArgumentParser(description='Process some model arguments.')
    parser.add_argument('echo', type=str, metavar='PATH', help='I/O location')
    parser.add_argument('-r', '--res', type=int, metavar='N', nargs='+', default=[20,20,50], help='Resolution in X,Y,Z directions')
    parser.add_argument('-v', '--verbose', action='store_true', required=False, default=False, help="Verbose output")
    parser.add_argument('--Tmin', type=float, required=False, default=298.0, help="Minimum temperature")
    parser.add_argument('--Tmax', type=float, required=False, default=500.0, help="Maximum temperature")
    args = parser.parse_args()


    data_dir = args.echo
    verbose  = args.verbose
    Tmin = args.Tmin
    Tmax = args.Tmax
    Nx, Ny, Nz = args.res # global size

else:
    data_dir = "../Data/"
    verbose  = True
    Tmin = 298.0
    Tmax = 500.0
    Nx, Ny, Nz = 20,20,50 # global size
# -

# ## Import geological surfaces

# +
with np.load(data_dir+"sydney_basin_surfaces.npz", "r") as npz:
    grid_list    = npz["layers"]
    grid_Xcoords = npz['Xcoords']
    grid_Ycoords = npz['Ycoords']

xmin, xmax = float(grid_Xcoords.min()), float(grid_Xcoords.max())
ymin, ymax = float(grid_Ycoords.min()), float(grid_Ycoords.max())
zmin, zmax = float(grid_list.min()),    float(grid_list.max())

if uw.mpi.rank == 0 and verbose:
    print("x {:8.3f} -> {:8.3f} km".format(xmin/1e3,xmax/1e3))
    print("y {:8.3f} -> {:8.3f} km".format(ymin/1e3,ymax/1e3))
    print("z {:8.3f} -> {:8.3f} km".format(zmin/1e3,zmax/1e3))


# set up interpolation object
interp = interpolate.RegularGridInterpolator((grid_Ycoords, grid_Xcoords), grid_list[0], method="nearest")

# update grid list for top and bottom of model
grid_list = list(grid_list)
grid_list.append(np.full_like(grid_list[0], zmin))
grid_list = np.array(grid_list)

n_layers = grid_list.shape[0]
# -

# ## Set up the mesh

# +
deformedmesh = True
elementType = "Q1"
mesh = uw.mesh.FeMesh_Cartesian( elementType = (elementType), 
                                 elementRes  = (Nx,Ny,Nz), 
                                 minCoord    = (xmin,ymin,zmin), 
                                 maxCoord    = (xmax,ymax,zmax)) 

gwPressureField            = mesh.add_variable( nodeDofCount=1 )
temperatureField           = mesh.add_variable( nodeDofCount=1 )
velocityField              = mesh.add_variable( nodeDofCount=3 )
heatProductionField        = mesh.add_variable( nodeDofCount=1 )


coords = mesh.data

Xcoords = np.unique(coords[:,0])
Ycoords = np.unique(coords[:,1])
Zcoords = np.unique(coords[:,2])
nx, ny, nz = Xcoords.size, Ycoords.size, Zcoords.size
# -

# ### Wrap mesh to surface topography
#  
# We want to deform z component so that we bunch up the mesh where we have valleys. The total number of cells should remain the same, only the vertical spacing should vary.

# +
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
# -


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

# +
topWall = mesh.specialSets["MaxK_VertexSet"]
bottomWall = mesh.specialSets["MinK_VertexSet"]

gwPressureBC = uw.conditions.DirichletCondition( variable      = gwPressureField, 
                                               indexSetsPerDof = ( topWall   ) )

temperatureBC = uw.conditions.DirichletCondition( variable        = temperatureField,
                                                  indexSetsPerDof = (topWall+bottomWall))


# +
# lower groundwater pressure BC - value is relative to gravity
maxgwpressure = 0.9

znorm = mesh.data[:,2].copy()
znorm -= zmin
znorm /= zmax
linear_gradient = 1.0 - znorm

initial_pressure = linear_gradient*maxgwpressure
initial_temperature = linear_gradient*(Tmax - Tmin) + Tmin
initial_temperature = np.clip(initial_temperature, Tmin, Tmax)


gwPressureField.data[:]  = initial_pressure.reshape(-1,1)
temperatureField.data[:] = initial_temperature.reshape(-1,1)

temperatureField.data[topWall] = Tmin
temperatureField.data[bottomWall] = Tmax
# -

# ## Set up the swarm particles
#
# It is best to set only one particle per cell, to prevent variations in hydaulic diffusivity within cells.

swarm         = uw.swarm.Swarm( mesh=mesh )
swarmLayout   = uw.swarm.layouts.PerCellGaussLayout(swarm=swarm,gaussPointCount=4)
swarm.populate_using_layout( layout=swarmLayout )

# +
materialIndex  = swarm.add_variable( dataType="int",    count=1 )
swarmVelocity  = swarm.add_variable( dataType="double", count=3 )

hydraulicDiffusivity    = swarm.add_variable( dataType="double", count=1 )
fn_hydraulicDiffusivity = swarm.add_variable( dataType="double", count=1 )
thermalDiffusivity      = swarm.add_variable( dataType="double", count=1 )
heatProduction          = swarm.add_variable( dataType="double", count=1 )
a_exponent              = swarm.add_variable( dataType="double", count=1 )
# -


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


# ### Assign material properties
#
# Use level sets to assign hydraulic diffusivities to a region on the mesh corresponding to any given material index.
#
# - $H$       : rate of heat production
# - $\rho$     : density
# - $k_h$     : hydraulic conductivity
# - $k_t$     : thermal conductivity
# - $\kappa_h$ : hydraulic diffusivity
# - $\kappa_t$ : thermal diffusivity
#
# First, there are some lithologies that need to be collapsed.

# +
def read_material_index(filename, cols):
    """
    Reads the material index with specified columns
    first 3 columns are ALWAYS:
    1. layer index
    2. material index
    3. material name
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

cols = [3,5,6,7,8,9,10,11]
layerIndex, matIndex, matName, [rho, kt0, dkt, H0, dH, kh0, dkh, a] = read_material_index(data_dir+"material_properties.csv", cols)

voxel_model_condensed = materialIndex.data.flatten()

# condense lith(s) to index(s)
for index in matIndex:
    for lith in layerIndex[index]:
        voxel_model_condensed[voxel_model_condensed == lith] = index

# +
interp.values = grid_list[0]
swarm_topography = interp((swarm.data[:,1],swarm.data[:,0]))

beta = 9.3e-3
depth = -1*(swarm.data[:,2] - zmax)
depth = -1*np.clip(swarm.data[:,2], zmin, 0.0)

# +
Storage = 1.
rho_water = 1000.

if deformedmesh:
    g = uw.function.misc.constant((0.,0.,-1.))
else:
    g = uw.function.misc.constant((0.,0.,0.))

gwPressureGrad = gwPressureField.fn_gradient

gMapFn = -g*rho_water*Storage
# -

# ## Load observations
#
# `evaluate_global()` raises an error if the coordinates are outside the domain. The next cells filter the well data.

# +
# load observations
well_E, well_N = np.loadtxt(data_dir+"well_ledger.csv", delimiter=',', usecols=(3,4), skiprows=1, unpack=True)
well_name = np.loadtxt(data_dir+"well_ledger.csv", delimiter=',', usecols=(0,), skiprows=1, dtype=str)

# filter observations outside domain
mask_wells = np.zeros_like(well_E, dtype=bool)
mask_wells += well_E < xmin
mask_wells += well_E > xmax
mask_wells += well_N < ymin
mask_wells += well_N > ymax
mask_wells = np.invert(mask_wells)

well_E = well_E[mask_wells]
well_N = well_N[mask_wells]
well_name = well_name[mask_wells]

# interpolate topography to well locations
interp.values = grid_list[0]
well_elevation = np.array(interp(np.c_[well_N, well_E]))

# +
mask_wells = np.zeros_like(well_E, dtype=bool)

misfit = np.array(0.0)
for i, name in enumerate(well_name):
    well_depth, well_temperature = np.loadtxt(data_dir+"well_data/{}.csv".format(name), delimiter=',',
                                              usecols=(0,2), skiprows=1, unpack=True)
    
    well_temperature = np.atleast_1d(well_temperature)
    well_temperature += 273.14 # convert to K
    well_xyz = np.empty((well_depth.size,3))
    well_xyz[:,0] = well_E[i]
    well_xyz[:,1] = well_N[i]
    well_xyz[:,2] = well_elevation[i] - well_depth ## subtract topography!
    
    exception = np.array(True)
    while exception:
        well_xyz = np.empty((well_depth.size,3))
        well_xyz[:,0] = well_E[i]
        well_xyz[:,1] = well_N[i]
        well_xyz[:,2] = well_elevation[i] - well_depth ## subtract topography!
        
        try:
            sim_temperature = temperatureField.evaluate_global(well_xyz)
            exception = np.array(False)
        except RuntimeError:
            exception = np.array(True)
        
        # only the root processor knows if this failed or not
        comm.Bcast([exception, MPI.BOOL], root=0)
        
        if exception:
            well_elevation[i] -= 10


# + active=""
# mask_wells = np.zeros_like(well_E, dtype=bool)
#
# misfit = np.array(0.0)
# for i, name in enumerate(well_name):
#     well_depth, well_temperature = np.loadtxt(data_dir+"well_data/{}.csv".format(name), delimiter=',',
#                                               usecols=(0,2), skiprows=1, unpack=True)
#     
#     well_temperature = np.atleast_1d(well_temperature)
#     well_temperature += 273.14 # convert to K
#     well_xyz = np.empty((well_depth.size,3))
#     well_xyz[:,0] = well_E[i]
#     well_xyz[:,1] = well_N[i]
#     well_xyz[:,2] = well_elevation[i] - well_depth - 100## subtract topography!
#
#     try:
#         sim_temperature = temperatureField.evaluate_global(well_xyz)
#         mask_wells[i] = True
#     except RuntimeError:
#         sim_temperature = np.array([0.0])
#         well_temperature = np.array([0.0])
#         mask_wells[i] = False
#         print(i, name)
#         
#     if uw.mpi.rank == 0:
#         sim_temperature = sim_temperature.ravel()
#         well_temperature[sim_temperature == 0] = 0.0 # protect against values that are above ground surface
#         misfit += ((well_temperature - sim_temperature)**2).sum()/len(sim_temperature)
#         
# # mask again
# well_E = well_E[mask_wells]
# well_N = well_N[mask_wells]
# well_name = well_name[mask_wells]
# -

# compare to observations
misfit = np.array(0.0)
for i, name in enumerate(well_name):
    well_depth, well_temperature = np.loadtxt(data_dir+"well_data/{}.csv".format(name), delimiter=',',
                                              usecols=(0,2), skiprows=1, unpack=True)

    well_temperature = np.atleast_1d(well_temperature)
    well_temperature += 273.14 # convert to K
    well_xyz = np.empty((well_depth.size,3))
    well_xyz[:,0] = well_E[i]
    well_xyz[:,1] = well_N[i]
    well_xyz[:,2] = well_elevation[i] - well_depth ## subtract topography!
    
    sim_temperature = temperatureField.evaluate_global(well_xyz)
    if uw.mpi.rank == 0:
        sim_temperature = sim_temperature.ravel()
        well_temperature[sim_temperature == 0] = 0.0 # protect against values that are above ground surface
        misfit += ((well_temperature - sim_temperature)**2).sum()/len(sim_temperature)


# +
def fn_kappa(k0, depth, beta):
    """ Wei et al. (1995) """
    return k0*(1.0 - depth/(58.0+1.02*depth))**3


def forward_model(x):
    """ 
    Variables in x:
    - k_h  : hydraulic conductivity
    - k_t  : thermal conductivity
    - H    : heat production
    - Tmax : bottom temperature BC
    """
    # unpack input vector
    kh, kt, H = np.array_split(x[:-1], 3)
    Tmax = x[-1]
    
    # initialise "default values"
    hydraulicDiffusivity.data[:] = kh[-1]
    thermalDiffusivity.data[:] = kt[-1]
    a_exponent.data[:] = a[-1]

    # populate mesh variables with material properties
    for i, index in enumerate(matIndex):
        mask_material = voxel_model_condensed == index
        hydraulicDiffusivity.data[mask_material] = kh[i]
        thermalDiffusivity.data[mask_material]   = kt[i]
        heatProduction.data[mask_material]       = H[i]
        a_exponent.data[mask_material]           = a[i]
        
    # project HP to mesh
    HPproj = uw.utils.MeshVariable_Projection(heatProductionField, heatProduction, swarm)
    HPproj.solve()
    
    # depth-dependent hydraulic conductivity
    #fn_hydraulicDiffusivity.data[:] = fn_kappa(hydraulicDiffusivity.data.ravel(), depth, beta).reshape(-1,1)
    zCoord = -(uw.function.input()[2] - zmax)
    kh_eff = hydraulicDiffusivity*(1.0 - zCoord/(58.0 + 1.02*zCoord))**3
    fn_hydraulicDiffusivity.data[:] = kh_eff.evaluate(swarm)
    # average out variation within a cell
    for cell in range(0, mesh.elementsLocal):
        mask_cell = swarm.owningCell.data == cell
        idx_cell  = np.nonzero(mask_cell)[0]
        fn_hydraulicDiffusivity.data[idx_cell] = fn_hydraulicDiffusivity.data[idx_cell].max()


    ## Set up groundwater equation
    if uw.mpi.rank == 0 and verbose:
        print("Solving grounwater equation...")
        print(kh)
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
    
    
    # temperature-dependent conductivity
    coeff = 1.0
    temperatureField.data[:] = np.clip(temperatureField.data, Tmin, Tmax)
    temperatureField.data[topWall] = Tmin
    temperatureField.data[bottomWall] = Tmax
    fn_thermalDiffusivity = thermalDiffusivity*(298.0/temperatureField)**a_exponent
    fn_source = uw.function.math.dot(-1.0*coeff*velocityField, temperatureField.fn_gradient) + heatProductionField
    
    ## Set up heat equation
    if uw.mpi.rank == 0 and verbose:
        print("Solving heat equation...")
    heateqn = uw.systems.SteadyStateHeat( temperatureField = temperatureField, \
                                          fn_diffusivity   = fn_thermalDiffusivity, \
                                          fn_heating       = fn_source, \
                                          conditions       = temperatureBC \
                                          )
    heatsolver = uw.systems.Solver(heateqn)
    heatsolver.solve(nonLinearIterate=True)

    
    # compare to observations
    misfit = np.array(0.0)
    for i, name in enumerate(well_name):
        well_depth, well_temperature = np.loadtxt(data_dir+"well_data/{}.csv".format(name), delimiter=',',
                                                  usecols=(0,2), skiprows=1, unpack=True)
        
        well_temperature = np.atleast_1d(well_temperature)
        well_temperature += 273.14 # convert to K
        well_xyz = np.empty((well_depth.size,3))
        well_xyz[:,0] = well_E[i]
        well_xyz[:,1] = well_N[i]
        well_xyz[:,2] = well_elevation[i]-well_depth

        sim_temperature = temperatureField.evaluate_global(well_xyz)
        if uw.mpi.rank == 0:
            sim_temperature = sim_temperature.ravel()
            well_temperature[sim_temperature == 0] = 0.0 # protect against values that are above ground surface
            # print(uw.mpi.rank, name, ((well_temperature - sim_temperature)**2).sum()/len(sim_temperature))
            misfit += ((well_temperature - sim_temperature)**2).sum()/len(sim_temperature)
    
    # compare priors
    if uw.mpi.rank == 0:
        # misfit += ((kh - kh0)**2/dkh**2).sum()
        misfit += ((kt - kt0)**2/dkt**2).sum()
        misfit += ((H - H0)**2/dH**2).sum()
    
    comm.Bcast([misfit, MPI.DOUBLE], root=0)
    
    if uw.mpi.rank == 0 and verbose:
        print("\n rank {} misfit = {}\n".format(uw.mpi.rank, misfit))
    
    return misfit


# +
# test forward model
x = np.hstack([kh0, kt0, H0, [Tmax]])
dx = 0.01*x

misfit = forward_model(x)
misfit = forward_model(x+dx)

# +
# define bounded optimisation
bounds_min = np.hstack([
    np.full_like(kh0, 1e-15),
    np.full_like(kt0, 0.05),
    np.zeros_like(H0),
    [298.]])
bounds_max = np.hstack([
    np.full_like(kh0, 0.001),
    np.full_like(kt0, 6.0),
    np.full_like(H0, 10e-6),
    [500+273.14]])

bounds = list(zip(bounds_min, bounds_max))

res = minimize(forward_model, x, method='TNC', bounds=bounds, options={'gtol': 1e-6, 'disp': True})
# -

well_velocity = velocityField.evaluate_global(np.column_stack([well_E, well_N, well_elevation]))
if uw.mpi.rank == 0:
    well_velocity_magnitude = np.hypot(*well_velocity.T)
    print("max velocity = {:.2e} m/day".format(well_velocity_magnitude.max() * 86400))
    # one would expect around 3 metres per day

# ## Save to HDF5

# +
xdmf_info_mesh  = mesh.save(data_dir+'mesh.h5')
xdmf_info_swarm = swarm.save(data_dir+'swarm.h5')

xdmf_info_matIndex = materialIndex.save(data_dir+'materialIndex.h5')
materialIndex.xdmf(data_dir+'materialIndex.xdmf', xdmf_info_matIndex, 'materialIndex', xdmf_info_swarm, 'TheSwarm')


# dummy mesh variable
phiField        = mesh.add_variable( nodeDofCount=1 )
heatflowField   = mesh.add_variable( nodeDofCount=3 )


# calculate heat flux
kTproj = uw.utils.MeshVariable_Projection(phiField, thermalDiffusivity, swarm)
kTproj.solve()

heatflowField.data[:] = temperatureField.fn_gradient.evaluate(mesh) * -phiField.data.reshape(-1,1)


rankField = mesh.add_variable( nodeDofCount=1 )
rankField.data[:] = uw.mpi.rank


for xdmf_info,save_name,save_object in [(xdmf_info_mesh, 'velocityField', velocityField),
                                        (xdmf_info_mesh, 'pressureField', gwPressureField),
                                        (xdmf_info_mesh, 'temperatureField', temperatureField),
                                        (xdmf_info_mesh, 'heatflowField', heatflowField),
                                        (xdmf_info_mesh, 'rankField', rankField),
                                        (xdmf_info_swarm, 'materialIndexSwarm', materialIndex),
                                        (xdmf_info_swarm, 'hydraulicDiffusivitySwarm', fn_hydraulicDiffusivity),
                                        (xdmf_info_swarm, 'thermalDiffusivitySwarm', thermalDiffusivity),
                                        (xdmf_info_swarm, 'heatProductionSwarm', heatProduction),
                                        ]:
    
    xdmf_info_var = save_object.save(data_dir+save_name+'.h5')
    save_object.xdmf(data_dir+save_name+'.xdmf', xdmf_info_var, save_name, xdmf_info, 'TheMesh')

    if save_name.endswith("Swarm"):
        # project swarm variables to the mesh
        hydproj = uw.utils.MeshVariable_Projection(phiField, save_object, swarm)
        hydproj.solve()

        field_name = save_name[:-5]+'Field'
        xdmf_info_var = phiField.save(data_dir+field_name+'.h5')
        phiField.xdmf(data_dir+field_name+'.xdmf', xdmf_info_var, field_name, xdmf_info_mesh, "TheMesh")
