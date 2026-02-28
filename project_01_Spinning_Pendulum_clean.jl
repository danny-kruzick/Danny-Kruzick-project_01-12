### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ 0a1b2c3d-0000-0000-0000-000000000001
begin
    # Packages used:
    # - Symbolics.jl (derive Euler–Lagrange equations)  https://docs.sciml.ai/Symbolics/stable/
    # - DifferentialEquations.jl / OrdinaryDiffEq.jl (solve ODEs) https://docs.sciml.ai/DiffEqDocs/stable/
    # - Plots.jl (plots + @gif animation) https://docs.juliaplots.org/stable/
    using Symbolics
    using DifferentialEquations
    using OrdinaryDiffEq
    using Plots
    import Latexify
end

# ╔═╡ 0a1b2c3d-0000-0000-0000-000000000002
md"""
# Project 01 — Spinning Pendulum

This notebook:
1. Derives the equation of motion with **least action** via the Lagrangian `L = T - V` (Symbolics).
2. Solves the motion for a **slow** and **fast** rotation rate `Ω` (DifferentialEquations).
3. Visualizes results with **plots** and an **animation** (Plots).

Model assumption (single-DOF):
- The support frame rotates about the **vertical z-axis** at constant rate `Ω`.
- The pendulum swings in the rotating frame plane (1 generalized coordinate `θ(t)`).
"""

# ╔═╡ 0a1b2c3d-0000-0000-0000-000000000003
begin
    # -----------------------
    # Parameters (edit here)
    # -----------------------
    g_val  = 9.81         # m/s^2
    L_val  = 0.15         # m
    w1_val = 0.10         # m (horizontal offset of pivot from spin axis)
    h1_val = 0.20         # m (pivot height for visualization only)
    m_val  = 0.10         # kg (kept; cancels in many cases but safe to include)

    # Initial conditions
    θ0 = 20 * pi/180      # rad
    ω0 = 0.0              # rad/s
    u0 = [θ0, ω0]

    # Time span
    tspan = (0.0, 10.0)

    # Rotation rates (slow / fast)
    Ω_slow = 2.0          # rad/s
    Ω_fast = 20.0         # rad/s
end

# ╔═╡ 0a1b2c3d-0000-0000-0000-000000000004
md"""
## Derive the equation of motion (Euler–Lagrange)

We write the bob position in inertial coordinates using cylindrical kinematics:

- `x = r cos(φ)`, `y = r sin(φ)`, `z = -L cos(θ)`
- `r = w1 + L sin(θ)`
- prescribed spin `φ = Ω t`

A useful identity is:

- `ẋ² + ẏ² = ṙ² + (r Ω)²`

This keeps the derivation compact and leads to a time-independent equation of motion for `θ(t)` when `Ω` is constant.
"""

# ╔═╡ 0a1b2c3d-0000-0000-0000-000000000005
begin
    # -----------------------
    # Symbolic derivation
    # -----------------------
    @variables t
    @variables m L g w1 Ω
    @variables θ(t)

    D = Differential(t)

    r  = w1 + L*sin(θ)
    z  = -L*cos(θ)

    ṙ = expand_derivatives(D(r))
    ż = expand_derivatives(D(z))

    T = (1//2) * m * (ṙ^2 + (r*Ω)^2 + ż^2)
    V = m * g * z
    Lag = T - V

    dL_dθdot = Symbolics.derivative(Lag, D(θ))
    dL_dθ    = Symbolics.derivative(Lag, θ)

    EL = expand_derivatives(D(dL_dθdot) - dL_dθ)

    # Solve for θ̈(θ,θ̇) and simplify
    θ̈_expr = solve_for(EL, D(D(θ)))
    θ̈_expr = simplify(θ̈_expr)

    # Build numerical function: θ̈ = f(θ, θ̇, g, L, w1, Ω, m)
    θ̈_func = build_function(θ̈_expr, θ, D(θ), g, L, w1, Ω, m; expression=Val(false))
end

# ╔═╡ 0a1b2c3d-0000-0000-0000-000000000006
md"""
### Equation of motion

Symbolics gives an explicit expression for the angular acceleration `θ̈ = f(θ, θ̇, …)`.

(Showing it here is optional, but useful for writeup / checking signs.)
"""

# ╔═╡ 0a1b2c3d-0000-0000-0000-000000000008
θ̈_expr

# ╔═╡ 0a1b2c3d-0000-0000-0000-000000000009
begin
    # -----------------------
    # ODE definition + solve
    # -----------------------
    function spinning_pendulum_ode!(du, u, p, t)
        θval = u[1]
        ωval = u[2]
        gval, Lval, w1val, Ωval, mval = p

        du[1] = ωval
        du[2] = θ̈_func(θval, ωval, gval, Lval, w1val, Ωval, mval)
        return nothing
    end

    p_slow = (g_val, L_val, w1_val, Ω_slow, m_val)
    p_fast = (g_val, L_val, w1_val, Ω_fast, m_val)

    prob_slow = ODEProblem(spinning_pendulum_ode!, u0, tspan, p_slow)
    prob_fast = ODEProblem(spinning_pendulum_ode!, u0, tspan, p_fast)

    sol_slow = solve(prob_slow, Tsit5(); reltol=1e-9, abstol=1e-9)
    sol_fast = solve(prob_fast, Tsit5(); reltol=1e-9, abstol=1e-9)
end

# ╔═╡ 0a1b2c3d-0000-0000-0000-00000000000a
md"""
## Plots: slow vs fast

We plot `θ(t)` and `ω(t)` for both rotation rates.
"""

# ╔═╡ 0a1b2c3d-0000-0000-0000-00000000000b
begin
    pltθ = plot(sol_slow.t, sol_slow[1, :],
        xlabel="t (s)", ylabel="θ (rad)", title="Angle vs Time",
        label="slow Ω=$(Ω_slow) rad/s")
    plot!(pltθ, sol_fast.t, sol_fast[1, :],
        label="fast Ω=$(Ω_fast) rad/s")

    pltω = plot(sol_slow.t, sol_slow[2, :],
        xlabel="t (s)", ylabel="ω (rad/s)", title="Angular Velocity vs Time",
        label="slow Ω=$(Ω_slow) rad/s")
    plot!(pltω, sol_fast.t, sol_fast[2, :],
        label="fast Ω=$(Ω_fast) rad/s")

    plot(pltθ, pltω, layout=(2,1), size=(900,700))
end

# ╔═╡ 0a1b2c3d-0000-0000-0000-00000000000c
md"""
## Kinematics for animation (inertial frame)

For visualization we compute bob position:
- `r = w1 + L sin(θ)`
- `φ = Ω t`
- `x = r cos φ`, `y = r sin φ`, `z = h1 - L cos θ`
"""

# ╔═╡ 0a1b2c3d-0000-0000-0000-00000000000d
begin
    function bob_position(sol, t; w1=w1_val, h1=h1_val, L=L_val, Ω=Ω_slow)
        θval = sol(t)[1]
        φ = Ω * t
        r = w1 + L*sin(θval)

        x = r*cos(φ)
        y = r*sin(φ)
        z = h1 - L*cos(θval)

        return x, y, z, r, θval
    end
end

# ╔═╡ 0a1b2c3d-0000-0000-0000-00000000000e
begin
    # Trajectory preview for slow case
    ts = range(tspan[1], tspan[2], length=400)

    xs = Float64[]; ys = Float64[]; zs = Float64[]; rs = Float64[]
    for tt in ts
        x,y,z,r,_ = bob_position(sol_slow, tt; Ω=Ω_slow)
        push!(xs,x); push!(ys,y); push!(zs,z); push!(rs,r)
    end

    p_top  = plot(xs, ys, xlabel="x (m)", ylabel="y (m)", aspect_ratio=:equal,
        title="Top view (slow Ω): bob trajectory", legend=false)

    p_side = plot(rs, zs, xlabel="r (m)", ylabel="z (m)",
        title="Side view (r,z) (slow Ω)", legend=false)

    plot(p_top, p_side, layout=(1,2), size=(900,400))
end

# ╔═╡ 0a1b2c3d-0000-0000-0000-00000000000f
md"""
## Animation (synced θ(t), ω(t), top view, side view)

Uses `@gif` from Plots.
"""

# ╔═╡ 0a1b2c3d-0000-0000-0000-000000000010
begin
    function animate_solution(sol; Ω=Ω_slow, labeltxt="slow", N=180)
        ts_anim = range(tspan[1], tspan[2], length=N)

        @gif for tt in ts_anim
            θval = sol(tt)[1]
            ωval = sol(tt)[2]

            pθ = plot(sol.t, sol[1,:], xlabel="t (s)", ylabel="θ (rad)",
                title="θ(t) — $(labeltxt) Ω=$(Ω) rad/s", legend=false)
            scatter!(pθ, [tt], [θval])

            pω = plot(sol.t, sol[2,:], xlabel="t (s)", ylabel="ω (rad/s)",
                title="ω(t) — $(labeltxt) Ω=$(Ω) rad/s", legend=false)
            scatter!(pω, [tt], [ωval])

            x,y,z,r,_ = bob_position(sol, tt; Ω=Ω)
            φ = Ω*tt
            xp = w1_val*cos(φ)
            yp = w1_val*sin(φ)

            p_top = plot(xlim=(-0.35,0.35), ylim=(-0.35,0.35),
                xlabel="x (m)", ylabel="y (m)", aspect_ratio=:equal,
                title="Top view (x-y)", legend=false)
            plot!(p_top, [0.0, xp], [0.0, yp], lw=2)
            plot!(p_top, [xp, x],  [yp, y],  lw=3)
            scatter!(p_top, [x], [y], ms=6)

            p_side = plot(xlim=(0.0, 0.35), ylim=(0.0, 0.35),
                xlabel="r (m)", ylabel="z (m)",
                title="Side view (r-z)", legend=false)
            plot!(p_side, [w1_val, r], [h1_val, z], lw=3)
            scatter!(p_side, [r], [z], ms=6)

            plot(pθ, pω, p_top, p_side, layout=(2,2), size=(900,700))
        end every 2
    end

    anim_slow = animate_solution(sol_slow; Ω=Ω_slow, labeltxt="slow")
end

# ╔═╡ 0a1b2c3d-0000-0000-0000-000000000011
animate_solution(sol_fast; Ω=Ω_fast, labeltxt="fast")

# ╔═╡ 0a1b2c3d-0000-0000-0000-000000000012
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
"""

# ╔═╡ 0a1b2c3d-0000-0000-0000-000000000013
PLUTO_MANIFEST_TOML_CONTENTS = """
# This field is intentionally left blank.
# Pluto will create a Manifest automatically when you open the notebook.
"""

# ╔═╡ Cell order:
# ╠═0a1b2c3d-0000-0000-0000-000000000001
# ╠═0a1b2c3d-0000-0000-0000-000000000002
# ╠═0a1b2c3d-0000-0000-0000-000000000003
# ╠═0a1b2c3d-0000-0000-0000-000000000004
# ╠═0a1b2c3d-0000-0000-0000-000000000005
# ╠═0a1b2c3d-0000-0000-0000-000000000006
# ╠═0a1b2c3d-0000-0000-0000-000000000008
# ╠═0a1b2c3d-0000-0000-0000-000000000009
# ╠═0a1b2c3d-0000-0000-0000-00000000000a
# ╠═0a1b2c3d-0000-0000-0000-00000000000b
# ╠═0a1b2c3d-0000-0000-0000-00000000000c
# ╠═0a1b2c3d-0000-0000-0000-00000000000d
# ╠═0a1b2c3d-0000-0000-0000-00000000000e
# ╠═0a1b2c3d-0000-0000-0000-00000000000f
# ╠═0a1b2c3d-0000-0000-0000-000000000010
# ╠═0a1b2c3d-0000-0000-0000-000000000011
# ╠═0a1b2c3d-0000-0000-0000-000000000012
# ╠═0a1b2c3d-0000-0000-0000-000000000013
