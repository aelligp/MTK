using Adapt

## Phase diagram
function init_rheologies(oxd_wt_sill, oxd_wt_host_rock; scaling = 1e0Pas, magma = true, CharDim = nothing)
    # Define parameters

    # sill_PD = "./Phase_diagrams/Heise_Sill.in"
    host_rock_PD = "./SillConvection/Phase_diagrams/Heise_Sill.in"
    sill_PD = "./SillConvection/Phase_diagrams/Heise_Host_rock.in"

    PD_Sill = MAGEMin_Diagram(sill_PD)
    # PD_Sill_GPU = Adapt.adapt(CuArray, PD_Sill)
    # PD_Sill_GPU = PD_Sill
    # host_rock_PD = "./Phase_diagrams/Heise_Host_rock.in"
    PD_Host_Rock = MAGEMin_Diagram(host_rock_PD)
    # PD_Host_Rock_GPU = Adapt.adapt(CuArray, PD_Host_Rock)
    # PD_Host_Rock_GPU = PD_Host_Rock
    sill = magma ? ViscosityPartialMelt_Costa_etal_2009(η = GiordanoMeltViscosity(oxd_wt = oxd_wt_sill, η0 = scaling)) : LinearViscous(η = 1.0e13Pa*s)
    host_rock = magma ? ViscosityPartialMelt_Costa_etal_2009(η = GiordanoMeltViscosity(oxd_wt = oxd_wt_host_rock, η0 = scaling)) : LinearViscous(η = 1.0e16Pa*s)


    # Define rheolgy struct
    rheology = (
        # Name              = "host_rock",
        SetMaterialParams(;
            Phase             = 1,
            Density           = ThreePhase_Density(ρmelt=Melt_DensityX(oxd_wt = oxd_wt_host_rock), ρsolid = T_Density(; ρ0 = 2700kg/m^3,α=3e-5/K), ρgas= RedlichKwong_Density()),
            # Density           = MeltDependent_Density(ρsolid=T_Density(; ρ0 = 2700kg/m^3,α=3e-5/K), ρmelt=Melt_DensityX(oxd_wt = oxd_wt_host_rock)),
            # Density           = PerpleX_LaMEM_Diagram(host_rock_PD),
            # Density           = PD_Host_Rock,
            HeatCapacity      = Latent_HeatCapacity(Cp=T_HeatCapacity_Whittington(), Q_L=350e3J/kg),
            # HeatCapacity      = T_HeatCapacity_Whittington(),
            # HeatCapacity      = ConstantHeatCapacity(),
            # HeatCapacity      = PD_Host_Rock_GPU,
            # Conductivity      = ConstantConductivity(; k = 3.0Watt/m/K),
            Conductivity      = T_Conductivity_Whittington(),
            CompositeRheology = CompositeRheology((host_rock,)),
            # Melting           = SmoothMelting(p=MeltingParam_Quadratic(T_s=(625+273)K,T_l=(875+273)K), k_liq=0.21/K),
            Melting           = MeltingParam_Volatile(),
            Solubility        = Liu2005_Solubility(),
            # Melting           = PerpleX_LaMEM_Diagram(host_rock_PD),
            # Melting           = PD_Host_Rock,
            Gravity           = ConstantGravity(),
            CharDim           = CharDim,
        ),
        # Name              = "Sill",
        SetMaterialParams(;
            Phase             = 2,
            Density           = ThreePhase_Density(ρmelt=Melt_DensityX(oxd_wt = oxd_wt_host_rock), ρsolid = T_Density(; ρ0 = 2700kg/m^3,α=3e-5/K), ρgas= RedlichKwong_Density()),
            # Density           = MeltDependent_Density(ρsolid=T_Density(; ρ0 = 2700kg/m^3,α=3e-5/K), ρmelt=Melt_DensityX(oxd_wt = oxd_wt_sill)),
            # Density           = PerpleX_LaMEM_Diagram(sill_PD),
            # Density           = PD_Sill,
            # HeatCapacity      = Latent_HeatCapacity(Cp=T_HeatCapacity_Whittington(), Q_L=350e3J/kg),
            HeatCapacity      = T_HeatCapacity_Whittington(),
            # HeatCapacity      = ConstantHeatCapacity(),
            # HeatCapacity      = PD_Sill_GPU,
            # Conductivity      = ConstantConductivity(; k = 3.0Watt/m/K),
            Conductivity      = T_Conductivity_Whittington(),
            CompositeRheology = CompositeRheology((sill,)),
            # Melting           = MeltingParam_Smooth3rdOrder(a=3043.0,b=-10552.0,c=12204.9,d=-4709.0),
            # Melting           = SmoothMelting(p=MeltingParam_Quadratic(T_s=(675+273)K,T_l=(1125+273)K), k_liq=0.21/K),
            Melting           = MeltingParam_Volatile(),
            Solubility        = Liu2005_Solubility(),
            # Melting           = PerpleX_LaMEM_Diagram(sill_PD),
            # Melting           = PD_Sill,
            CharDim           = CharDim,
        ),
    )
    return rheology
end

function init_phases!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    return @parallel (@idx ni) _init_phases!(phases, phase_grid, particles.coords, particles.index, xvi)
end

@parallel_indices (I...) function _init_phases!(phases, phase_grid, pcoords::NTuple{N, T}, index, xvi) where {N, T}

    ni = size(phases)

    for ip in cellaxes(phases)
        # quick escape
        @index(index[ip, I...]) == 0 && continue

        pᵢ = ntuple(Val(N)) do i
            @index pcoords[i][ip, I...]
        end

        d = Inf # distance to the nearest particle
        particle_phase = -1
        for offi in 0:1, offj in 0:1
            ii = I[1] + offi
            jj = I[2] + offj

            !(ii ≤ ni[1]) && continue
            !(jj ≤ ni[2]) && continue

            xvᵢ = (
                xvi[1][ii],
                xvi[2][jj],
            )
            d_ijk = √(sum((pᵢ[i] - xvᵢ[i])^2 for i in 1:N))
            if d_ijk < d
                d = d_ijk
                particle_phase = phase_grid[ii, jj]
            end
        end
        @index phases[ip, I...] = Float64(particle_phase)
    end

    return nothing
end
