#!/usr/bin/env python
# coding: utf-8
# # 5 - Optimisation: groundwater + heat flow

# +
import numpy as np
import os
import csv
import argparse
from time import time
from scipy import interpolate
from scipy.spatial import cKDTree
from scipy import optimize
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
    parser.add_argument('-s', '--surrogate', action='store_true', required=False, default=False, help="Use surrogate information if available")
    args = parser.parse_args()


    data_dir = args.echo
    verbose  = args.verbose
    surrogate = args.surrogate
    Tmin = args.Tmin
    Tmax = args.Tmax
    Nx, Ny, Nz = args.res # global size

else:
    data_dir = "../Data/"
    verbose  = True
    surrogate= False
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
temperatureField0          = mesh.add_variable( nodeDofCount=1 )
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

sealevel = 0.0
seafloor = topWall[mesh.data[topWall,2] < sealevel]

gwPressureField.data[topWall] = 0.
gwPressureField.data[seafloor] = -((mesh.data[seafloor,2]-sealevel)*1.0).reshape(-1,1)
temperatureField.data[topWall] = Tmin
temperatureField.data[bottomWall] = Tmax
# -

# ### Import pressure BC

# +
with np.load(data_dir+"water_table_surface.npz") as npz:
    wt = npz['z']
    wt_x = npz['x']
    wt_y = npz['y']

rgi_wt = interpolate.RegularGridInterpolator((wt_y, wt_x), wt)

wt_interp = rgi_wt(mesh.data[topWall,0:2][:,::-1])
gwPressureField.data[topWall] = (-wt_interp * 1000.0 * 9.81).reshape(-1,1)
# -

# ## Set up the swarm particles
#
# It is best to set only one particle per cell, to prevent variations in hydaulic diffusivity within cells.

swarm         = uw.swarm.Swarm( mesh=mesh )
swarmLayout   = uw.swarm.layouts.PerCellGaussLayout(swarm=swarm,gaussPointCount=4)
swarm.populate_using_layout( layout=swarmLayout )

# +
materialIndex  = swarm.add_variable( dataType="int",    count=1 )
cellCentroid   = swarm.add_variable( dataType="double", count=3 )
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
    cellCentroid.data[idx_cell] = cell_centroid


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
swarm_topography = interp((cellCentroid.data[:,1],cellCentroid.data[:,0]))

beta = 9.3e-3
depth = -1.0*(cellCentroid.data[:,2] - swarm_topography)
depth = np.clip(depth, 0.0, zmax-zmin)

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

# ## Load solvers
#
# Significant time savings can be had by loading the solvers for groundwater flow and heat flow prior to the forward model.

# +
# initialise "default values"
fn_hydraulicDiffusivity.data[:] = kh0[-1]
hydraulicDiffusivity.data[:] = kh0[-1]
thermalDiffusivity.data[:] = kt0[-1]
a_exponent.data[:] = a[-1]

coeff = 1.0
fn_thermalDiffusivity = thermalDiffusivity*(298.0/temperatureField)**a_exponent
fn_source = uw.function.math.dot(-1.0*coeff*velocityField, temperatureField.fn_gradient) + heatProductionField

# +
# groundwater solver
gwadvDiff = uw.systems.SteadyStateDarcyFlow(
                                            velocityField    = velocityField, \
                                            pressureField    = gwPressureField, \
                                            fn_diffusivity   = fn_hydraulicDiffusivity, \
                                            conditions       = [gwPressureBC], \
                                            fn_bodyforce     = -gMapFn, \
                                            voronoi_swarm    = swarm, \
                                            swarmVarVelocity = swarmVelocity)
gwsolver = uw.systems.Solver(gwadvDiff)

# heatflow solver
heateqn = uw.systems.SteadyStateHeat( temperatureField = temperatureField, \
                                      fn_diffusivity   = fn_thermalDiffusivity, \
                                      fn_heating       = fn_source, \
                                      conditions       = temperatureBC \
                                      )
heatsolver = uw.systems.Solver(heateqn)
# -

# ## Load observations
#
# `evaluate_global()` raises an error if the coordinates are outside the domain. The next cells filter the well data.

# +
# load observations
well_E, well_N, well_dTdz = np.loadtxt(data_dir+"well_ledger.csv", delimiter=',', usecols=(3,4,9),
                                       skiprows=1, unpack=True)

well_dTdz[np.isnan(well_dTdz)] = 0

# filter observations outside domain
mask_wells = np.zeros_like(well_E, dtype=bool)
mask_wells += well_E < xmin
mask_wells += well_E > xmax
mask_wells += well_N < ymin
mask_wells += well_N > ymax
mask_wells += well_dTdz <= 0.0
mask_wells += well_dTdz > 0.15
mask_wells = np.invert(mask_wells)

well_E = well_E[mask_wells]
well_N = well_N[mask_wells]
well_dTdz = well_dTdz[mask_wells]

# interpolate topography to well locations
interp.values = grid_list[0]
well_elevation = np.array(interp(np.c_[well_N, well_E]))

well_xyz = np.c_[well_E, well_N, well_elevation]

nwells = well_xyz.shape[0]
print("number of well observations = {}".format(nwells))
# -

for i in range(0, nwells):
    exception = np.array(True)
    while exception:
        try:
            sim_temperature = temperatureField.evaluate_global(np.atleast_2d(well_xyz[i]))
            if sim_temperature.ravel() > 0:
                exception = np.array(False)
        except:
            pass
        
        # only the root processor knows if this failed or not
        comm.Bcast([exception, MPI.BOOL], root=0)
        
        if exception:
            well_xyz[i,2] -= 10

# Recharge rates from [Crosbie _et al._ (2018)](https://doi.org/10.1016/j.jhydrol.2017.08.003)

# +
recharge_E, recharge_N, recharge_vel, recharge_vel_std = np.loadtxt(data_dir+"recharge_Crosbie2018.csv",
                                                                    delimiter=',', unpack=True, usecols=(2,3,4,5))
# convert mm/yr to m/s
recharge_vel *= 3.17097919838e-11
recharge_vel_std *= 3.17097919838e-11

mask_recharge = np.zeros_like(recharge_E, dtype=bool)
mask_recharge += recharge_E < xmin
mask_recharge += recharge_E > xmax
mask_recharge += recharge_N < ymin
mask_recharge += recharge_N > ymax
mask_recharge += recharge_vel_std <= 0.0
mask_recharge = np.invert(mask_recharge)

recharge_E       = recharge_E[mask_recharge]
recharge_N       = recharge_N[mask_recharge]
recharge_vel     = recharge_vel[mask_recharge]
recharge_vel_std = recharge_vel_std[mask_recharge]

recharge_Z = np.array(interp(np.c_[recharge_N, recharge_E]))

recharge_xyz = np.c_[recharge_E, recharge_N, recharge_Z]


# +
velocityField.data[:] = 1.0
hyp = np.sqrt(3.0)

for i in range(0, recharge_xyz.shape[0]):
    exception = np.array(True)
    while exception:
        try:
            sim_velocity = velocityField.evaluate_global(np.atleast_2d(recharge_xyz[i]))
            sim_velocity_mag = np.sqrt((sim_velocity**2).sum())

            if np.allclose(sim_velocity_mag, hyp):
                exception = np.array(False)
        except:
            pass

        # only the root processor knows if this failed or not
        comm.Bcast([exception, MPI.BOOL], root=0)

        if exception:
            recharge_xyz[i,2] -= 10
# -

# Hydraulic pressure

# +
gw_data = np.loadtxt(data_dir+'NGIS_groundwater_levels_SGB.csv', delimiter=',', usecols=(4,5,7,8,9,10), skiprows=1)
gw_E, gw_N, gw_Z, gw_depth, gw_level, gw_level_std = gw_data.T

gw_xyz = np.c_[gw_E, gw_N, gw_Z]

gw_pressure = (gw_depth - gw_level)*1000.0*9.81
gw_pressure_std = gw_level_std*1000.0*9.81
gw_pressure_std[gw_pressure_std == 0] = np.percentile(gw_level_std, 99)*1000.0*9.81


for i in range(0, gw_data.shape[0]):
    exception = np.array(True)
    while exception:
        try:
            sim_pressure = gwPressureField.evaluate_global(np.atleast_2d(gw_xyz[i]))
            if sim_pressure.ravel() > 0:
                exception = np.array(False)
        except:
            pass

        # only the root processor knows if this failed or not
        comm.Bcast([exception, MPI.BOOL], root=0)

        if exception:
            gw_xyz[i,2] -= 10

gw_xyz[:,2] -= gw_depth


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
    ti = time()
    
    # check we haven't already got a solution
    dist, idx = mintree.query(x)

    if dist == 0.0 and surrogate:
        misfit = minimiser_misfits[idx]
        if verbose:
            print("using surrogate model, misfit = {}".format(misfit))
        return misfit
    else:
        # unpack input vector
        kh, kt, H = np.array_split(x[:-1], 3)
        Tmax = x[-1]
        
        # scale variables
        kh = 10.0**kh # log10 scale
        H  = H*1e-6 # convert to micro

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
        fn_hydraulicDiffusivity.data[:] = fn_kappa(hydraulicDiffusivity.data.ravel(), depth, beta).reshape(-1,1)


        ## Set up groundwater equation
        if uw.mpi.rank == 0 and verbose:
            print("Solving grounwater equation...")
        gwsolver.solve()


        # temperature-dependent conductivity
        temperatureField.data[:] = np.clip(temperatureField.data, Tmin, Tmax)
        temperatureField.data[topWall] = Tmin
        temperatureField.data[bottomWall] = Tmax

        ## Set up heat equation
        if uw.mpi.rank == 0 and verbose:
            print("Solving heat equation...")
        for its in range(0, 10):
            temperatureField0.data[:] = temperatureField.data[:]
            heatsolver.solve(nonLinearIterate=False)

            Tdiff = np.array(np.abs(temperatureField0.data[:] - temperatureField.data[:]).max())
            Tdiff_all = np.array(0.0)
            comm.Allreduce([Tdiff, MPI.DOUBLE], [Tdiff_all, MPI.DOUBLE], op=MPI.MAX)
            if Tdiff_all < 0.01:
                break

        # compare to observations
        misfit = np.array(0.0)
        sim_dTdz = temperatureField.fn_gradient[2].evaluate_global(well_xyz)
        if uw.mpi.rank == 0:
            sim_dTdz = -1.0*sim_dTdz.ravel()
            misfit += ((well_dTdz - sim_dTdz)**2/0.1**2).sum()
            print(((well_dTdz - sim_dTdz)**2/0.1**2).sum())
            
        sim_vel = velocityField.evaluate_global(recharge_xyz)
        if uw.mpi.rank == 0:
            sim_vel_mag = np.sqrt((sim_vel**2).sum(axis=1))
            misfit += (((np.log10(recharge_vel) - np.log10(sim_vel_mag))**2)/np.log10(recharge_vel_std)**2).sum()
            print((((np.log10(recharge_vel) - np.log10(sim_vel_mag))**2)/np.log10(recharge_vel_std)**2).sum())

        sim_pressure = gwPressureField.evaluate_global(gw_xyz)
        if uw.mpi.rank == 0:
            sim_pressure = sim_pressure.ravel()
            misfit += ((gw_pressure - sim_pressure)**2/gw_pressure_std**2).sum()
            print(((gw_pressure - sim_pressure)**2/gw_pressure_std**2).sum())

        # compare priors
        if uw.mpi.rank == 0:
            misfit += ((np.log10(kh) - np.log10(kh0))**2).sum()
            misfit += ((kt - kt0)**2/dkt**2).sum()
            misfit += ((H - H0)**2/dH**2).sum()

        comm.Bcast([misfit, MPI.DOUBLE], root=0)

        if uw.mpi.rank == 0:
            with open(data_dir+'minimiser_results.csv', 'a') as f:
                rowwriter = csv.writer(f, delimiter=',')
                rowwriter.writerow(np.hstack([[misfit], x]))

            if verbose:
                print("\n rank {} in {:.2f} sec misfit = {}\n".format(uw.mpi.rank, time()-ti, misfit))

        return misfit
# -


x = np.hstack([np.log10(kh0), kt0, H0*1e6, [Tmax]])
dx = 0.01*x

# ## Initialise output table
#
# A place to store misfit and $x$ parameters.

# +
import os
if "minimiser_results.csv" in os.listdir(data_dir):
    # load existing minimiser results table
    minimiser_results_data = np.loadtxt(data_dir+"minimiser_results.csv", delimiter=',', )
    if not len(minimiser_results_data):
        minimiser_results_data = np.zeros((1,x.size+1))
    minimiser_results = minimiser_results_data[:,1:]
    minimiser_misfits = minimiser_results_data[:,0]
else:
    minimiser_results = np.zeros((1,x.size))
    minimiser_misfits = np.array([0.0])
    if uw.mpi.rank == 0:
        with open(data_dir+'minimiser_results.csv', 'w') as f:
            pass
    
mintree = cKDTree(minimiser_results)

# +
# test forward model
fm0 = forward_model(x)
fm1 = forward_model(x+dx)
print("finite difference = {}".format(fm1-fm0))


# define bounded optimisation
bounds_lower = np.hstack([
    np.full_like(kh0, -13),
    np.full_like(kt0, 0.05),
    np.zeros_like(H0),
    [298.]])
bounds_upper = np.hstack([
    np.full_like(kh0, -3),
    np.full_like(kt0, 6.0),
    np.full_like(H0, 10),
    [600+273.14]])

bounds = list(zip(bounds_lower, bounds_upper))


finite_diff_step = np.hstack([np.full_like(kh0, 0.1), np.full_like(kt0, 0.01), np.full_like(H0, 0.01), [1.0]])

def obj_func(x):
    return forward_model(x)
def obj_grad(x):
    return optimize.approx_fprime(x, forward_model, finite_diff_step)  



# + active=""
# (2556, 34) <-- for TNC. 1 day on HPC with 768 cores yields (159, 34), so it would take 14 days to find inv sol.
# BUT TNC fails to converge because the line search is crap.

# +
res = optimize.differential_evolution(forward_model, bounds=bounds, popsize=2, seed=42, disp=True)
print(res)

if uw.mpi.rank == 0:
    np.savez_compressed(data_dir+"optimisation_result.npz", **res)

# + active=""
# res = optimize.shgo(forward_model, bounds=bounds,)
# print(res)
# -

# **steepest descent**
#
# ```python
# for i in range(0, 100):
#     grad_f = optimize.approx_fprime(x, forward_model, finite_diff_step)
#     mu = 0.01
#     #search_gradient = np.full_like(x, -0.1)
#     #mu, fc, gc, new_fval, old_fval, new_slope = optimize.line_search(obj_func, obj_grad, x, search_gradient,grad_f)
#     x = x - mu*grad_f
#     x = np.clip(x, bounds_lower, bounds_upper)
# ```

# **minimize**
#
# ```python
# np.random.seed(42)
# options={'gtol': 1e-6, 'disp': True, 'finite_diff_rel_step': finite_diff_step}
# res = optimize.minimize(forward_model, x, method='TNC', bounds=bounds, options=options)
# print(res)
#
# if uw.mpi.rank == 0:
#     np.savez_compressed(data_dir+"optimisation_result.npz",
#                         x       = res.x,
#                         fun     = res.fun,
#                         success = res.success,
#                         status  = res.status,
#                         nfev    = res.nfev,
#                         nit     = res.nit,
#                         #hessian = res.hess_inv.todense()
#                         )
# ```

# **EMCEE (MCMC sampler)**
#
# ```python
# import emcee
# np.random.seed(42)
#
# nwalker=x.size*2
#
# x0 = np.empty((nwalker, x.size))
# x0[0] = x
# for i in range(1, nwalker):
#     x0[i] = np.random.uniform(bounds_lower, bounds_upper)
#
# sampler = emcee.EnsembleSampler(nwalker, x.size, forward_model,)
# state = sampler.run_mcmc(x0, 10, skip_initial_state_check=True) # burn in
#
# sampler.reset()
# sampler.run_mcmc(state, 100)
#
# posterior = sampler.get_chain(flat=True)
# np.savetxt("mcmc_posterior.txt", posterior)
# print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
# ```

# **PINNUTS (Hamiltonian Monte Carlo)**
#
# ```python
# from pinnuts import pinnuts
#
# def obj_fun_grad(x):
#     logp = obj_func(x)
#     grad = obj_grad(x)
#     return logp, grad
#
# np.random.seed(1)
# samples, lnprob, epsilon = pinnuts(obj_fun_grad, 50, 10, x, finite_diff_step)
# print("samples\n", samples, "lnprob\n", lnprob, "epsilon\n", epsilon)
# ```

# + active=""
# well_velocity = velocityField.evaluate_global(np.column_stack([well_E, well_N, well_elevation]))
# if uw.mpi.rank == 0:
#     well_velocity_magnitude = np.hypot(*well_velocity.T)
#     print("max velocity = {:.2e} m/day".format(well_velocity_magnitude.max() * 86400))
#     # one would expect around 3 metres per day
# -

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
thermalDiffusivity.data[:] = fn_thermalDiffusivity.evaluate(swarm)
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
# -

# ## Save minimiser results


