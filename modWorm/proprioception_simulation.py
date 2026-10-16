"""
modWorm: Modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

__author__ = 'Jimin Kim: jk55@u.washington.edu'

import numpy as np

from scipy import integrate, interpolate
from collections import namedtuple

from modWorm import network_interactions as n_inter
from modWorm import body_simulations as b_sim
from modWorm import utils

from modWorm import Main
from modWorm import require_julia
from modWorm.numerics import NumericalIntegrationError, validate_ode_step

#####################################################################################################################################################
# MASTER FUNCTIONS ##################################################################################################################################
#####################################################################################################################################################

def run_network(NN_nv, NN_mb, input_mat, extv_mat = False, stiffness_mb = False, interp_method = 'linear', 
                rtol_nv = 5e-6, atol_nv = 1e-3, rtol_mb = 2.5e-6, atol_mb = 1e-3):

    integration_prep = prep_integration(NN_nv, NN_mb, input_mat, extv_mat, interp_method)
    r_nv, r_mb = configure_ode_solver(NN_nv, NN_mb, stiffness_mb, rtol_nv, atol_nv, rtol_mb, atol_mb)
    solution = integrate_ode(NN_nv, NN_mb, r_nv, r_mb, integration_prep)

    return solution

#####################################################################################################################################################
# PREPARATION FUNCTIONS #############################################################################################################################
#####################################################################################################################################################

def prep_integration(NN_nv, NN_mb, input_mat, extv_mat, interp_method):

    # Time array for the solution
    nsteps = len(input_mat)
    tf = (nsteps - 1) * NN_nv.timescale
    timepoints = np.linspace(0, tf, nsteps)

    # Solution placeholders for nervous system states
    v_solution = np.full((nsteps, NN_nv.network_Size), np.nan)
    s_solution = np.full((nsteps, NN_nv.network_Size), np.nan)
    vthmat = np.full((nsteps, NN_nv.network_Size), np.nan)

    # Solution placeholders for muscles and body states
    muscle_force = np.full((nsteps, NN_mb.body_SegCount * 2), np.nan)
    b_solution = np.full((nsteps, 2+NN_mb.body_SegCount), np.nan)

    # Interpolate input and external voltage matrix
    NN_nv.interp_i_Ext = interpolate.interp1d(timepoints, input_mat, axis=0, kind = interp_method, fill_value = "extrapolate")

    if type(extv_mat) != bool:

        NN_nv.interp_v_Ext = interpolate.interp1d(timepoints, extv_mat, axis=0, kind = interp_method, fill_value = "extrapolate")

    NN_nv.id, NN_mb.id = 0, 0

    integration_prep = {"nsteps": nsteps,
                        "input_mat": input_mat,
                        "v_solution": v_solution,
                        "s_solution": s_solution,
                        "vthmat": vthmat,
                        "muscle_force": muscle_force,
                        "b_solution": b_solution,
                        "timescale": NN_nv.timescale}

    integration_prep = namedtuple('integration_prep', integration_prep.keys())(*integration_prep.values())

    return integration_prep

def configure_ode_solver(NN_nv, NN_mb, stiffness_mb, rtol_nv, atol_nv, rtol_mb, atol_mb):

    if 'forward_network_Jacobian' in dir(NN_nv):

        r_nv = integrate.ode(NN_nv.forward_Network, NN_nv.forward_network_Jacobian).set_integrator('vode', rtol = rtol_nv, atol = atol_nv, method = 'bdf')

    else:

        r_nv = integrate.ode(NN_nv.forward_Network).set_integrator('vode', rtol = rtol_nv, atol = atol_nv, method = 'bdf')

    if stiffness_mb == False:

        r_mb = integrate.ode(NN_mb.forward_Body).set_integrator('dopri5', rtol = rtol_mb, atol = atol_mb)

    else:

        # scipy.integrate.ode does not provide a Radau integrator; use its
        # supported stiff VODE/BDF implementation for the legacy step API.
        r_mb = integrate.ode(NN_mb.forward_Body).set_integrator(
            'vode', rtol=rtol_mb, atol=atol_mb, method='bdf'
        )

    r_nv.set_initial_value(NN_nv.initcond, 0)
    r_mb.set_initial_value(NN_mb.initcond, 0)

    return r_nv, r_mb

def integrate_ode(NN_nv, NN_mb, r_nv, r_mb, integration_prep):

    integration_prep.v_solution[0, :] = NN_nv.initcond[:NN_nv.network_Size]
    integration_prep.s_solution[0, :] = NN_nv.initcond[NN_nv.network_Size:NN_nv.network_Size*2]
    integration_prep.b_solution[0, :] = NN_mb.initcond[:2+NN_mb.body_SegCount]
    pdf_modulation = getattr(NN_nv, 'pdfr1_modulation', None)
    pdf_signal_solution = None
    if pdf_modulation is not None:
        pdf_signal_solution = np.zeros(
            (integration_prep.nsteps, len(pdf_modulation.target_indices))
        )
        pdf_signal_solution[0, :] = pdf_modulation.signal_state

    k = 0

    while k < integration_prep.nsteps - 1:
        if not r_nv.successful() or not r_mb.successful():
            raise NumericalIntegrationError(
                f"Proprioceptive integration failed before sample {k} "
                f"(neural t={r_nv.t}, body t={r_mb.t})"
            )

        # update vth
        integration_prep.vthmat[k, :] = NN_nv.compute_Vth(integration_prep.input_mat[k])

        # update muscle forces
        muscle_force_k = NN_mb.forward_Muscles(integration_prep.v_solution[k, :], integration_prep.vthmat[k, :])
        integration_prep.muscle_force[k, :] = muscle_force_k
        NN_mb.fR, NN_mb.fL = muscle_force_k[:NN_mb.body_SegCount-1], muscle_force_k[NN_mb.body_SegCount:-1]

        # compute proprioceptive fdb input
        NN_nv.input_FDB = NN_nv.proprioceptive_FDB(NN_mb, integration_prep.v_solution, integration_prep.s_solution, integration_prep.vthmat,
                                                   integration_prep.muscle_force, integration_prep.b_solution, k)

        # integrate nv and mb
        if pdf_modulation is not None:
            pdf_signal_solution[k + 1, :] = pdf_modulation.advance(
                NN_nv.timescale, r_nv.t
            )
        target_time = (k + 1) * NN_nv.timescale
        r_nv.integrate(target_time)
        neural_state = validate_ode_step(
            r_nv, target_time, len(NN_nv.initcond),
            operation="neural integration", step_index=k + 1,
        )
        r_mb.integrate(target_time)
        body_state = validate_ode_step(
            r_mb, target_time, len(NN_mb.initcond),
            operation="body integration", step_index=k + 1,
        )

        # update v, s, and body locs
        integration_prep.v_solution[k+1, :] = neural_state[:NN_nv.network_Size]
        integration_prep.s_solution[k+1, :] = neural_state[NN_nv.network_Size:NN_nv.network_Size*2]
        integration_prep.b_solution[k+1, :] = body_state[:2+NN_mb.body_SegCount]

        k += 1

    integration_prep.vthmat[-1, :] = NN_nv.compute_Vth(integration_prep.input_mat[-1])
    integration_prep.muscle_force[-1, :] = NN_mb.forward_Muscles(
        integration_prep.v_solution[-1, :], integration_prep.vthmat[-1, :]
    )
    if not (
        np.isfinite(integration_prep.v_solution).all()
        and np.isfinite(integration_prep.s_solution).all()
        and np.isfinite(integration_prep.vthmat).all()
        and np.isfinite(integration_prep.muscle_force).all()
        and np.isfinite(integration_prep.b_solution).all()
    ):
        raise NumericalIntegrationError(
            "Proprioceptive integration left one or more trajectory samples uncomputed"
        )

    x0, y0 = integration_prep.b_solution[:, 0], integration_prep.b_solution[:, 1]
    phi = integration_prep.b_solution[:, 2:26]

    x, y = b_sim.solve_xy(NN_mb, x0, y0, phi)
    x_post, y_post = b_sim.postprocess_xy(NN_mb, x, y)

    solution = {"v_solution" : integration_prep.v_solution,
                "s_solution": integration_prep.s_solution,
                "v_threshold" : integration_prep.vthmat,
                "muscle_force" : integration_prep.muscle_force,
                "x_solution" : x_post,
                "y_solution" : y_post,
                "phi": phi,
                "NN_nv_id": NN_nv.id,
                "NN_mb_id": NN_mb.id}
    if pdf_signal_solution is not None:
        solution["pdf_signal_solution"] = pdf_signal_solution
    return solution

#####################################################################################################################################################
# MASTER FUNCTIONS (Julia) ##########################################################################################################################
#####################################################################################################################################################

# Native Julia storage for Python-provided NumPy arrays.
# Sundials must not initialize its solution cache with PythonCall.PyArray.
# Arrays are copied ONCE at the inter-language boundary; scalar settings
# and the existing modWorm functions remain unchanged.
Main.eval("""
function modworm_native_ppc_dict(raw)
    native = Dict{String,Any}()
    for (key, value) in raw
        native[string(key)] = value isa AbstractArray ? Array(value) : value
    end
    return native
end
function modworm_native_ppc_dict_list(raws)
    return [modworm_native_ppc_dict(raw) for raw in raws]
end
""")

def run_network_julia_ensemble(NN_nv_ens, NN_mb_ens, input_mat_ens, extv_mat_ens = False, stiffness_mb = False, maxiters_mb = 1e6, verbose_period = False, 
                               rtol_nv = 5e-6, atol_nv = 1e-3, rtol_mb = 2.5e-6, atol_mb = 1e-3):

    require_julia()
    NN_nv_ens[0].forward_Network()
    NN_nv_ens[0].compute_Vth()
    NN_nv_ens[0].proprioceptive_FDB()
    NN_mb_ens[0].forward_Muscles()
    NN_mb_ens[0].forward_Body()

    vars_NN_nv_ens, vars_NN_mb_ens = [], []

    for ens_k in range(len(NN_nv_ens)):

        NN_nv, NN_mb = NN_nv_ens[ens_k], NN_mb_ens[ens_k]
        input_mat = utils.array_or_list(input_mat_ens, ens_k)

        nsteps = len(input_mat)
        tf = (nsteps - 1) * NN_nv.timescale
        timepoints = np.linspace(0, tf, nsteps)
    
        vars_NN_nv = vars(NN_nv)
        vars_NN_nv["nsteps"] = nsteps
        vars_NN_nv["tf"] = tf
        vars_NN_nv["timepoints"] = timepoints
        vars_NN_nv["input_mat"] = input_mat
        vars_NN_nv["input_FDB"] = np.zeros(NN_nv.network_Size)
        vars_NN_nv["id"] = ens_k

        if type(extv_mat_ens) != bool:

            extv_mat = utils.array_or_list(extv_mat_ens, ens_k)
            assert len(extv_mat) == len(input_mat)

            vars_NN_nv["extv_mat"] = extv_mat

        vars_NN_mb = vars(NN_mb)
        vars_NN_mb["timescale"] = NN_nv.timescale
        vars_NN_mb["fR"] = np.zeros(NN_mb.body_SegCount-1)
        vars_NN_mb["fL"] = np.zeros(NN_mb.body_SegCount-1)
        vars_NN_mb["stiffness"] = stiffness_mb
        vars_NN_mb["maxiters"] = int(maxiters_mb)
        vars_NN_mb["id"] = ens_k

        vars_NN_nv_ens.append(vars_NN_nv)
        vars_NN_mb_ens.append(vars_NN_mb)

    ppc_solution_dict_ens = Main.run_network_ppc_ensemble(
        Main.modworm_native_ppc_dict_list(vars_NN_nv_ens),
        Main.modworm_native_ppc_dict_list(vars_NN_mb_ens),
        verbose_period, rtol_nv, atol_nv, rtol_mb, atol_mb
    )
    ppc_solution_dict_ens = utils.sort_process_ensemble_sols(ppc_solution_dict_ens, NN_mb_ens)

    return ppc_solution_dict_ens

def run_network_julia(NN_nv, NN_mb, input_mat, extv_mat = False, stiffness_mb = False, maxiters_mb = 1e6, verbose_period = False, 
                      rtol_nv = 5e-6, atol_nv = 1e-3, rtol_mb = 2.5e-6, atol_mb = 1e-3):

    require_julia()
    NN_nv.forward_Network()
    NN_nv.compute_Vth()
    NN_nv.proprioceptive_FDB()
    NN_mb.forward_Muscles()
    NN_mb.forward_Body()

    nsteps = len(input_mat)
    tf = (nsteps - 1) * NN_nv.timescale
    timepoints = np.linspace(0, tf, nsteps)
    
    vars_NN_nv = vars(NN_nv)
    vars_NN_nv["nsteps"] = nsteps
    vars_NN_nv["tf"] = tf
    vars_NN_nv["timepoints"] = timepoints
    vars_NN_nv["input_mat"] = input_mat
    vars_NN_nv["input_FDB"] = np.zeros(NN_nv.network_Size)
    vars_NN_nv["id"] = 0

    if type(extv_mat) != bool:

        vars_NN_nv["extv_mat"] = extv_mat

    vars_NN_mb = vars(NN_mb)
    vars_NN_mb["timescale"] = NN_nv.timescale
    vars_NN_mb["fR"] = np.zeros(NN_mb.body_SegCount-1)
    vars_NN_mb["fL"] = np.zeros(NN_mb.body_SegCount-1)
    vars_NN_mb["stiffness"] = stiffness_mb
    vars_NN_mb["maxiters"] = int(maxiters_mb)
    vars_NN_mb["id"] = 0

    ppc_solution_dict = Main.run_network_ppc(
        Main.modworm_native_ppc_dict(vars_NN_nv),
        Main.modworm_native_ppc_dict(vars_NN_mb),
        verbose_period, rtol_nv, atol_nv, rtol_mb, atol_mb
    )

    x0 = np.asarray(ppc_solution_dict["x0"], dtype=float)
    y0 = np.asarray(ppc_solution_dict["y0"], dtype=float)
    phi = np.asarray(ppc_solution_dict["phi"], dtype=float)
    x, y = b_sim.solve_xy(NN_mb, x0, y0, phi)
    x_post, y_post = b_sim.postprocess_xy(NN_mb, x, y)

    ppc_solution_dict["x_solution"] = x_post
    ppc_solution_dict["y_solution"] = y_post

    return ppc_solution_dict

Main.eval("""

    function run_network_ppc_ensemble(NN_nv_ens, NN_mb_ens, verbose_period, rtol_nv, atol_nv, rtol_mb, atol_mb)

        GC.gc()

        NN_nv_ens, NN_mb_ens = construct_NN_ppc_ensemble(NN_nv_ens, NN_mb_ens)

        integration_prep_ens = prep_network_integration_ppc_ensemble(NN_nv_ens, NN_mb_ens)
        integrator_nv_ens, integrator_mb_ens = configure_ode_solver_ppc_ensemble(NN_nv_ens, NN_mb_ens, integration_prep_ens, rtol_nv, atol_nv, rtol_mb, atol_mb)

        ppc_solution_dict_ens = []

        Threads.@threads for ens_k = 1:size(NN_nv_ens)[1]

            ppc_solution_dict = integrate_ode_ppc(NN_nv_ens[ens_k], NN_mb_ens[ens_k], integration_prep_ens[ens_k], integrator_nv_ens[ens_k], integrator_mb_ens[ens_k], verbose_period)

            push!(ppc_solution_dict_ens, ppc_solution_dict)

            end
        
        return ppc_solution_dict_ens

    end

    function run_network_ppc(NN_nv, NN_mb, verbose_period, rtol_nv, atol_nv, rtol_mb, atol_mb)

        GC.gc()

        NN_nv, NN_mb = construct_NN_ppc(NN_nv, NN_mb)

        integration_prep = prep_network_integration_ppc(NN_nv, NN_mb)
        integrator_nv, integrator_mb = configure_ode_solver_ppc(NN_nv, NN_mb, integration_prep, rtol_nv, atol_nv, rtol_mb, atol_mb)

        ppc_solution_dict = integrate_ode_ppc(NN_nv, NN_mb, integration_prep, integrator_nv, integrator_mb, verbose_period)
        
        return ppc_solution_dict

    end

    """)

#####################################################################################################################################################
# PREPARATION FUNCTIONS (Julia) #####################################################################################################################
#####################################################################################################################################################

Main.eval("""

    function construct_NN_ppc_ensemble(NN_nv_ens, NN_mb_ens)

        NN_nv_nt_ens, NN_mb_nt_ens = [], []

        for ens_k = 1:size(NN_nv_ens)[1]

            NN_nv, NN_mb = construct_NN_ppc(NN_nv_ens[ens_k], NN_mb_ens[ens_k])

            push!(NN_nv_nt_ens, NN_nv)
            push!(NN_mb_nt_ens, NN_mb)

        end

        return NN_nv_nt_ens, NN_mb_nt_ens

    end

    function construct_NN_ppc(NN_nv, NN_mb)

        NN_nv["interp_i_Ext"] = LinearInterpolation(NN_nv["timepoints"], [NN_nv["input_mat"][i,:] for i in 1:size(NN_nv["input_mat"], 1)], extrapolation_bc = Line())

        if haskey(NN_nv, "extv_mat") == true

            NN_nv["interp_v_Ext"] = LinearInterpolation(NN_nv["timepoints"], [NN_nv["extv_mat"][i,:] for i in 1:size(NN_nv["extv_mat"], 1)], extrapolation_bc = Line())

        end

        NN_nv = (; (Symbol(k) => v for (k,v) in NN_nv)...)
        NN_mb = (; (Symbol(k) => v for (k,v) in NN_mb)...)

        return NN_nv, NN_mb

    end

#----------------------------------------------------------------------------------------------------------------------------------------------------

    function prep_network_integration_ppc_ensemble(NN_nv_ens, NN_mb_ens)

        integration_prep_ens = []

        for ens_k = 1:size(NN_nv_ens)[1]

            integration_prep = prep_network_integration_ppc(NN_nv_ens[ens_k], NN_mb_ens[ens_k])

            push!(integration_prep_ens, integration_prep)

        end

        return integration_prep_ens

    end

    function prep_network_integration_ppc(NN_nv, NN_mb)

        v_solution = fill(NaN, NN_nv.nsteps, NN_nv.network_Size)
        s_solution = fill(NaN, NN_nv.nsteps, NN_nv.network_Size)
        vthmat = fill(NaN, NN_nv.nsteps, NN_nv.network_Size)

        muscle_force = fill(NaN, NN_nv.nsteps, NN_mb.body_SegCount * 2)
        b_solution = fill(NaN, NN_nv.nsteps, 2+NN_mb.body_SegCount)

        return (input_mat = NN_nv.input_mat,
                nsteps = NN_nv.nsteps,
                tf = NN_nv.tf,
                v_solution = v_solution,
                s_solution = s_solution,
                vthmat = vthmat,
                muscle_force = muscle_force,
                b_solution = b_solution)    

    end

#----------------------------------------------------------------------------------------------------------------------------------------------------

    function configure_ode_solver_ppc_ensemble(NN_nv_ens, NN_mb_ens, integration_prep_ens, rtol_nv, atol_nv, rtol_mb, atol_mb)

        integrator_nv_ens, integrator_mb_ens = [], []

        for ens_k = 1:size(NN_nv_ens)[1]

            integrator_nv, integrator_mb = configure_ode_solver_ppc(NN_nv_ens[ens_k], NN_mb_ens[ens_k], integration_prep_ens[ens_k], rtol_nv, atol_nv, rtol_mb, atol_mb)

            push!(integrator_nv_ens, integrator_nv)
            push!(integrator_mb_ens, integrator_mb)

        end

        return integrator_nv_ens, integrator_mb_ens

    end

    function configure_ode_solver_ppc(NN_nv, NN_mb, integration_prep, rtol_nv, atol_nv, rtol_mb, atol_mb)  

        r_nv = ODEProblem(forward_Network!, NN_nv.initcond, (0, integration_prep.tf), NN_nv)
        r_mb = ODEProblem(forward_Body!, NN_mb.initcond, (0, integration_prep.tf), NN_mb)

        integrator_nv = init(r_nv, CVODE_BDF(), reltol = rtol_nv, abstol = atol_nv, save_everystep = false)

        if NN_mb.stiffness == true

            integrator_mb = init(r_mb, CVODE_BDF(), reltol = rtol_mb, abstol = atol_mb, save_everystep = false, maxiters = NN_mb.maxiters)

        else

            integrator_mb = init(r_mb, Tsit5(), reltol = rtol_mb, abstol = atol_mb, save_everystep = false, maxiters = NN_mb.maxiters)

        end

        return integrator_nv, integrator_mb

    end

#----------------------------------------------------------------------------------------------------------------------------------------------------

    function integrate_ode_ppc(NN_nv, NN_mb, integration_prep, integrator_nv, integrator_mb, verbose_period)

        integration_prep.v_solution[1, :] = NN_nv.initcond[1:NN_nv.network_Size]
        integration_prep.s_solution[1, :] = NN_nv.initcond[NN_nv.network_Size+1:NN_nv.network_Size*2]
        integration_prep.b_solution[1, :] = NN_mb.initcond[1:2+NN_mb.body_SegCount]

        milestone = Int(round(integration_prep.nsteps * verbose_period))

        k = 1

        while k < integration_prep.nsteps

            integration_prep.vthmat[k, :] = compute_Vth(NN_nv, integration_prep.input_mat[k, :])

            muscle_force_k = forward_Muscles(NN_mb, integration_prep.v_solution[k, :], integration_prep.vthmat[k, :])
            integration_prep.muscle_force[k, :] = muscle_force_k
            NN_mb.fR[:], NN_mb.fL[:] = muscle_force_k[1:NN_mb.body_SegCount-1], muscle_force_k[NN_mb.body_SegCount+1:end-1]

            NN_nv.input_FDB[:] .= proprioceptive_FDB(NN_nv, NN_mb, integration_prep.v_solution, integration_prep.s_solution, integration_prep.vthmat, 
                                                 integration_prep.muscle_force, integration_prep.b_solution, k)

            step!(integrator_nv, NN_nv.timescale, true)
            step!(integrator_mb, NN_mb.timescale, true)
            expected_time = k * NN_nv.timescale
            isapprox(integrator_nv.t, expected_time; rtol=1e-10, atol=1e-12) ||
                error("neural integrator stopped at t=$(integrator_nv.t), expected t=$expected_time")
            isapprox(integrator_mb.t, expected_time; rtol=1e-10, atol=1e-12) ||
                error("body integrator stopped at t=$(integrator_mb.t), expected t=$expected_time")
            all(isfinite, integrator_nv.u) ||
                error("neural integrator returned a non-finite state at sample $(k+1)")
            all(isfinite, integrator_mb.u) ||
                error("body integrator returned a non-finite state at sample $(k+1)")

            integration_prep.v_solution[k+1, :] = integrator_nv.u[1:NN_nv.network_Size]
            integration_prep.s_solution[k+1, :] = integrator_nv.u[NN_nv.network_Size+1:NN_nv.network_Size*2]
            integration_prep.b_solution[k+1, :] = integrator_mb.u[1:2+NN_mb.body_SegCount]

            if verbose_period != 0

                print_progress(NN_nv.id, k, milestone, integration_prep.nsteps, verbose_period)

            end

            k += 1

        end

        integration_prep.vthmat[end, :] = compute_Vth(NN_nv, integration_prep.input_mat[end, :])
        integration_prep.muscle_force[end, :] = forward_Muscles(
            NN_mb, integration_prep.v_solution[end, :], integration_prep.vthmat[end, :]
        )
        all(isfinite, integration_prep.v_solution) &&
            all(isfinite, integration_prep.s_solution) &&
            all(isfinite, integration_prep.vthmat) &&
            all(isfinite, integration_prep.muscle_force) &&
            all(isfinite, integration_prep.b_solution) ||
            error("proprioceptive integration left uncomputed or non-finite samples")

        x0, y0 = integration_prep.b_solution[:, 1], integration_prep.b_solution[:, 2]
        phi = integration_prep.b_solution[:, 3:end]

        return Dict([("v_solution", integration_prep.v_solution)
                     ("s_solution", integration_prep.s_solution)
                     ("v_threshold", integration_prep.vthmat)
                     ("muscle_force", integration_prep.muscle_force)
                     ("x0", x0)
                     ("y0", y0)
                     ("phi", phi)
                     ("NN_nv_id", NN_nv.id)
                     ("NN_mb_id", NN_mb.id)])

    end

    """)