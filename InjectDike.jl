
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)using Parameters
using JustPIC
using JustRelax
using Printf, Statistics, LinearAlgebra, GeoParams, GLMakie, CellArrays
using StaticArrays
using ImplicitGlobalGrid, MPI: MPI
const backend = CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

PS_Setup(:Threads, Float64, 2)            # initialize parallel stencil in 2D
environment!(model)

## function from MTK
@with_kw struct Dike    # stores info about dike
    # Note: since we utilize the "Parameters.jl" package, we can add more keywords here w/out breaking the rest of the code
    #
    # We can also define only a few parameters here (like Q and ΔP) and compute Width/Thickness from that
    # Or we can define thickness
    Angle       ::  Vector{Float64} =   [0.]                                  # Strike/Dip angle of dike
    Type        ::  String          =   "SquareDike"                          # Type of dike
    T           ::  Float64         =   950.0                                 # Temperature of dike
    E           ::  Float64         =   1.5e10                                # Youngs modulus (only required for elastic dikes)
    ν           ::  Float64         =   0.3                                   # Poison ratio of host rocks
    ΔP          ::  Float64         =   1e6;                                  # Overpressure of elastic dike
    Q           ::  Float64         =   1000;                                 # Volume of elastic dike
    W           ::  Float64         =   (3*E*Q/(16*(1-ν^2)*ΔP))^(1.0/3.0);    # Width of dike/sill
    H           ::  Float64         =   8*(1-ν^2)*ΔP*W/(π*E);                 # (maximum) Thickness of dike/sill
    Center      ::  Vector{Float64} =   [20e3 ; -10e3]                        # Center
    Phase       ::  Int64           =   2;                                    # Phase of newly injected magma
end


## function from MTK
function DisplacementAroundPennyShapedDike(dike::Dike, CartesianPoint::SVector, dim)

    # extract required info from dike structure
    (;ν,E,W, H) = dike;

    Displacement = Vector{Float64}(undef, dim);

    # Compute r & z; note that the Sun solution is defined for z>=0 (vertical)
    if      dim==2; r = sqrt(CartesianPoint[1]^2);                       z = abs(CartesianPoint[2]);
    elseif  dim==3; r = sqrt(CartesianPoint[1]^2 + CartesianPoint[2]^2); z = abs(CartesianPoint[3]); end

    if r==0; r=1e-3; end

    B::Float64   =  H;                          # maximum thickness of dike
    a::Float64   =  W/2.0;                      # radius

    # note, we can either specify B and a, and compute pressure p and injected volume Q
    # Alternatively, it is also possible to:
    #       - specify p and a, and compute B and Q
    #       - specify volume Q & p and compute radius and B
    #
    # What is best to do is to be decided later (and doesn't change the code below)
    Q   =   B*(2pi*a.^2)/3.0;               # volume of dike (follows from eq. 9 and 10a)
    p   =   3E*Q/(16.0*(1.0 - ν^2)*a^3);    # overpressure of dike (from eq. 10a) = 3E*pi*B/(8*(1-ν^2)*a)

    # Compute displacement, using complex functions
    R1  =   sqrt(r^2. + (z - im*a)^2);
    R2  =   sqrt(r^2. + (z + im*a)^2);

    # equation 7a:
    U   =   im*p*(1+ν)*(1-2ν)/(2pi*E)*( r*log( (R2+z+im*a)/(R1 +z- im*a))
                                                - r/2*((im*a-3z-R2)/(R2+z+im*a)
                                                + (R1+3z+im*a)/(R1+z-im*a))
                                                - (2z^2 * r)/(1 -2ν)*(1/(R2*(R2+z+im*a)) -1/(R1*(R1+z-im*a)))
                                                + (2*z*r)/(1-2ν)*(1/R2 - 1/R1) );
    # equation 7b:
    W   =       2*im*p*(1-ν^2)/(pi*E)*( z*log( (R2+z+im*a)/(R1+z-im*a))
                                                - (R2-R1)
                                                - 1/(2*(1-ν))*( z*log( (R2+z+im*a)/(R1+z-im*a)) - im*a*z*(1/R2 + 1/R1)) );

    # Displacements are the real parts of U and W.
    #  Note that this is the total required elastic displacement (in m) to open the dike.
    #  If we only want to open the dike partially, we will need to normalize these values accordingly (done externally)
    Uz   =  real(W);  # vertical displacement should be corrected for z<0
    Ur   =  real(U);
    if (CartesianPoint[end]<0); Uz = -Uz; end
    if (CartesianPoint[1]  <0); Ur = -Ur; end

    if      dim==2
        Displacement = [Ur;Uz]
    elseif  dim==3
        # Ur denotes the radial displacement; in 3D we have to decompose this in x and y components
        x   = abs(CartesianPoint[1]); y = abs(CartesianPoint[2]);
        Ux  = x/r*Ur; Uy = y/r*Ur;

        Displacement = [Ux;Uy;Uz]
    end

    return Displacement, B, p
end


#to be renamed
#should calculate the displacement per particle
function move_particles!(particles, x, y)
    (; coords, index) = particles
    px, py = coords
    ni     = size(px)

    @parallel_indices (I...) function move_particles!(px, py, index, x, y)

        for ip in JustRelax.cellaxes(px)
            !(JustRelax.@cell(index[ip, I...])) && continue
            px_ip = JustRelax.@cell   px[ip, I...]
            py_ip = JustRelax.@cell   py[ip, I...]

            displacement, Bmax = DisplacementAroundPennyShapedDike(dike, SA[px_ip,  py_ip],2)
            displacement .= displacement/Bmax

            JustRelax.@cell   px[ip, I...] = px_ip  + displacement[1]
            JustRelax.@cell   py[ip, I...] = py_ip  + displacement[2]
        end
        return nothing
    end

    @parallel (@idx ni) move_particles!(coords..., index, x, y)

end

#get rid of @simd
function  RotatePoints_2D!(Xrot,Zrot, X,Z, RotMat)
    @simd for i in eachindex(X) # linear indexing
        pt_rot      =   RotMat*SA[X[i], Z[i]];
        Xrot[i]     =   pt_rot[1]
        Zrot[i]     =   pt_rot[2]
    end
end


function Inject_Dike_particles(xvi, dike::Dike, particles)
    dim = length(xvi)

    if dim ==2
        x = [x for x in xvi[1], y in xvi[2]]
        y = [y for x in xvi[1], y in xvi[2]]

        (;Angle, Type, Center, H, W) = dike
        α = Angle[1]

        RotMat = SA[cos(α) -sin(α); sin(α) cos(α)]
        x = x.-Center[1]
        y = y.-Center[2]
        RotatePoints_2D!(x,y, x,y, RotMat)
        move_particles!(particles, x, y)
        inject = check_injection(particles)
        # inject && inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer,), xvi)

        # RotatePoints_2D!(Vx, Vy, Vx_rot, Vy_rot, RotMat')
        RotatePoints_2D!(x,y, x,y, RotMat')
        x = x.+Center[1]
        y = y.+Center[2]

        # move_particles!(particles, Displacement)
        println("I'm groot")
    end
    return x, y

end

## for plotting the particles
function plot_particles(particles, pPhases)
    p = particles.coords
    # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
    ppx, ppy = p
    # pxv = ustrip.(dimensionalize(ppx.data[:], km, CharDim))
    # pyv = ustrip.(dimensionalize(ppy.data[:], km, CharDim))
    pxv = ppx.data[:]
    pyv = ppy.data[:]
    clr = pPhases.data[:]
    # clrT = pT.data[:]
    idxv = particles.index.data[:]
    f,ax,h=scatter(Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma)
    Colorbar(f[1,2], h)
    f
end


## not working function based only on particles
# function Inject_Dike_particles_new(dike::Dike, particles)
#     (; coords, index) = particles
#     px, py = coords
#     ni     = size(px)
#     # dim = length(ni)
#     if length(ni) == 2
#         @parallel_indices (I...) function Inject_Dike_particles_new(px, py, index, dike)
#             for ip in JustRelax.cellaxes(px)
#                 !(JustRelax.@cell(index[ip, I...])) && continue
#                 px_ip = JustRelax.@cell   px[ip, I...]
#                 py_ip = JustRelax.@cell   py[ip, I...]

#                 (;Angle, Type, Center, H, W) = dike
#                 α = Angle[1]
#                 Δ = H

#                 RotMat = SA[cos(α) -sin(α); sin(α) cos(α)]
#                 px_ip = px_ip.-Center[1]
#                 py_ip = py_ip.-Center[2]
#                 # x = x.-Center[1]
#                 # y = y.-Center[2]
#                 # RotatePoints_2D!(x,y, x,y, RotMat)
#                 RotatePoints_2D!(px_ip,py_ip, px_ip,py_ip, RotMat)
#                 displacement, Bmax = DisplacementAroundPennyShapedDike(dike, SA[px_ip,  py_ip],2)
#                 displacement .= displacement/Bmax

#                 RotatePoints_2D!(px_ip,py_ip, px_ip,py_ip, RotMat')
#                 px_ip = px_ip.+Center[1]
#                 py_ip = py_ip.+Center[2]
#             end
#             # RotatePoints_2D!(x,y, x,y, RotMat')
#             # x = x.+Center[1]
#             # y = y.+Center[2]

#             # move_particles!(particles, Displacement)
#             # println("I'm groot")

#             return nothing
#         end
#         @parallel (@idx ni) Inject_Dike_particles_new(particles.coords..., particles.index, dike)
#     end

# end

###################################
#MWE
###################################
#-------JustRelax parameters-------------------------------------------------------------

ar = 1 # aspect ratio
n = 64
nx = n * ar - 2
ny = n - 2
nz = n - 2
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI=true)...)
else
    igg
end

# Domain setup for JustRelax
lx, lz          = 10e3, 10e3 # nondim if CharDim=CharDim
li              = lx, lz
b_width         = (4, 4, 0) #boundary width
origin          = 0.0, -lz
igg             = igg
di              = @. li / ni # grid step in x- and y-direction
# di = @. li / (nx_g(), ny_g()) # grid step in x- and y-direction
grid            = Geometry(ni, li; origin = origin)
(; xci, xvi)    = grid # nodes at the center and vertices of the cells
#----------------------------------------------------------------------
nxcell, max_xcell, min_xcell = 20, 40, 5 # nxcell = initial number of particles per cell; max_cell = maximum particles accepted in a cell; min_xcell = minimum number of particles in a cell
particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, xvi..., di..., ni...
)
# velocity grids
grid_vx, grid_vy = velocity_grids(xci, xvi, di)
# temperature
pT, pPhases      = init_cell_arrays(particles, Val(3))
particle_args = (pT, pPhases)

W_in, H_in              = 5e3,1e3; # Width and thickness of dike
T_in                    = 1000
H_ran, W_ran            = (xvi[1].stop, xvi[2].start).* [0.2;0.3];                          # Size of domain in which we randomly place dikes and range of angles
Dike_Type               = "ElasticDike"
# Tracers                 = StructArray{Tracer}(undef, 1);
nTr_dike                = 300;
cen              = ((xvi[1].stop, xvi[2].stop).+(xvi[1].start,xvi[2].start))./2.0 .+ rand(-0.5:1e-3:0.5, 2).*[W_ran;H_ran];
#   Dike injection based on Toba seismic inversion
if cen[end] < ustrip(nondimensionalize(-5km, CharDim))
    Angle_rand = [rand(80.0:0.1:100.0)] # Orientation: near-vertical @ depth
else
    Angle_rand = [rand(-10.0:0.1:10.0)] # Orientation: near-vertical @ shallower depth
end

dike      =   Dike(Angle=Angle_rand, W=W_in, H=H_in, Type=Dike_Type, T=T_in, Center=cen, #=ΔP=10e6,=#Phase =2 );

x,y = Inject_Dike_particles(xvi,dike, particles)
