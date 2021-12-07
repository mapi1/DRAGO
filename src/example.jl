# Recreate figure 3.3 from the original paper
using Main.DRAGO
using Plots
using LinearAlgebra
using Distributions
using BandedMatrices
using BenchmarkTools
N = 1000
t = range(0, 55, length = N)
f = -sin.(t) .- cos.(0.5t)
(y, f, x, w, S) = simulateData(f, percGaps = 0.45, maxGapRatio = 0.02, outlierStrength = 1.5)
present = [1:N;][diag(S' * S) .== 1]

(fRes, xRes, residual) = drago(y, 0.6, 500, S, ΔFmin = 10e-2)

# L1 norm
norm1(x) = sum(abs, x)

# L2 norm
norm2(x) = sqrt(sum(abs2, x))


# Plotting
begin
    p1 = plot(f, lab = "Simulated")
    p2 = scatter(present, S * y, lab = "Noisy")
    p3 = scatter(present, S * y, lab = "", color = :lightgrey, alpha = 0.2)
    rmse = norm2(f .- fRes) / norm2(f) 
    plot!(fRes, color = :black, linewidth = 2, lab = "Smoothed signal with RMSE: $(round(rmse, digits = 2))")
    p4 = plot(x[present], color = :red, lab = "True outlier")
    plot!(xRes, color = :black, lab = "Estimated outlier")
    p5 = plot(residual, lab = "Residual", color = :black)
    plot(p1, p2, p3, p4, p5, ylims = (-3, 3), layout = (5, 1), size = (800, 800))
end

N = 5
Dn = Tridiagonal(ones(N-2), -2ones(N-1), ones(N-2))
Dn' * Dn
d = copy(Dn[2:end-1, 1:end])

Dn[1,1] = -1
Dn[end,end] = -1
Dn' * Dn

d' * d

BandedMatrix((0 => ones(N-2), 1 => -2ones(N-2), 2 => ones(N-2)), (N-2, N))
@benchmark (fRes, xRes, residual) = drago(y, 0.6, 500, S, ΔFmin = 10e-2)
@profview (fRes, xRes, residual) = drago(y, 0.6, 500, S, ΔFmin = 10e-2)