# Julia accelerator for modWorm's existing Python-configured PPC model.
# Loaded only when the user selects the optional Julia backend.
# Reuses fwd_i_Linear, fwd_i_Electrical, fwd_i_Chemical,
# fwd_activity_Chemical, compute_Vth, the existing Julia body RHS,
# construct_NN_ppc, prep_network_integration_ppc, and integrate_ode_ppc.
#
# Controls are right-continuous and piecewise constant at the supplied
# simulation timepoints. The PDF signal is evaluated analytically between
# control knots, including at the internal solver RHS evaluation times.

function modworm_neural_modulated!(du, u, p, t)
    n = p.network_Size
    v = @view u[1:n]
    s = @view u[n+1:2n]
    # Original PPC is a delayed projected VOLTAGE OFFSET, not a current.
    v_effective = v .+ p.input_FDB
    i_ext = p.interp_i_Ext(t)
    i_linear = fwd_i_Linear(p, v_effective)
    i_electrical = fwd_i_Electrical(p, v_effective)
    i_chemical = fwd_i_Chemical(p, v_effective, s)

    # n >= 2; the last interval also defines the limiting value at tf.
    k = clamp(searchsortedlast(p.timepoints, t), 1, p.nsteps-1)
    a_mod1 = p.modworm_5ht_activation[k]
    i_mod1 = (p.modworm_g_mod1 .* a_mod1) .* (v_effective .- p.modworm_e_cl)

    i_pdfr1 = zeros(Float64, n)
    if p.modworm_pdf_gain != 0.0 && length(p.modworm_pdf_targets) > 0
        dt = clamp(t - p.timepoints[k], 0.0, p.timepoints[k+1] - p.timepoints[k])
        decay = exp(-dt / p.modworm_pdf_tau)
        target = p.modworm_pdf_equilibria[k]
        for (j, neuron_id) in enumerate(p.modworm_pdf_targets)
            state = target + (p.modworm_pdf_knots[k,j] - target) * decay
            i_pdfr1[neuron_id] = p.modworm_pdf_gain * state
        end
    end
    # Signs match the Python neural RHS: MOD-1 subtracts; PDF bias adds.
    du[1:n] .= (-i_linear .- i_electrical .- i_chemical .-
                 i_mod1 .+ i_pdfr1 .+ i_ext) ./ p.neuron_C
    du[n+1:2n] .= fwd_activity_Chemical(
        p, v_effective, s, compute_Vth(p, i_ext)
    )
    return nothing
end

function modworm_run_modulated_ppc(
    raw_nn, raw_mb, reltol_nv, abstol_nv, reltol_mb, abstol_mb,
    verbose_period
)
    # Original modWorm model preparation: no duplicate equations for body,
    # neural threshold, muscle mapping, or delayed PPC projection.
    # Copy NumPy-backed PyArray values into native Julia arrays once,
    # before ODEProblem is constructed.  This is required for Sundials'
    # solution-cache type consistency (Vector{Float64}, not PyArray).
    nn, mb = construct_NN_ppc(
        modworm_native_ppc_dict(raw_nn),
        modworm_native_ppc_dict(raw_mb),
    )
    nn.initcond isa Vector{Float64} || error("Neural initial state must be a Julia Vector{Float64}")
    mb.initcond isa Vector{Float64} || error("Body initial state must be a Julia Vector{Float64}")
    prep = prep_network_integration_ppc(nn, mb)
    neural_problem = ODEProblem(
        modworm_neural_modulated!, nn.initcond, (0.0, prep.tf), nn
    )
    body_problem = ODEProblem(forward_Body!, mb.initcond, (0.0, prep.tf), mb)

    # Stop precisely on all prescribed control boundaries. The legacy
    # integrate_ode_ppc function advances these integrators as persistent
    # objects, rather than repeatedly constructing a BDF solver per frame.
    neural = init(
        neural_problem, CVODE_BDF(), reltol=reltol_nv, abstol=abstol_nv,
        save_everystep=false,
        tstops=collect(nn.timepoints[2:end-1])
    )
    if mb.stiffness
        body = init(
            body_problem, CVODE_BDF(), reltol=reltol_mb,
            abstol=abstol_mb, save_everystep=false, maxiters=mb.maxiters
        )
    else
        body = init(
            body_problem, Tsit5(), reltol=reltol_mb,
            abstol=abstol_mb, save_everystep=false, maxiters=mb.maxiters
        )
    end
    return integrate_ode_ppc(nn, mb, prep, neural, body, verbose_period)
end
