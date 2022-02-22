# Recreate figure 3.3 from the original paper
using Plots
using DRAGO

t = 0:0.05:100
s = -sin.(t) .- cos.(0.5t)
(y, x, w, S) = simulate_data(s, percentage_gaps = 0.45, max_gap_ratio = 0.025)
(fRes, xRes, residual) = drago(y, 0.6, 500, S, ΔFmin = 10e-2)

p1 = plot(s, lab = "Simulated")
p2 = scatter(get_present(S), S * y, lab = "Noise + Gaps (y)")
p3 = scatter(get_present(S), S * y, lab = "", color = :lightgrey, alpha = 0.2)
plot!(fRes, color = :black, linewidth = 2, lab = "Smoothed signal (f)")
p4 = plot(x, color = :red, lab = "True outlier")
plot!(S' * xRes, color = :black, lab = "Estimated outlier (x)")
p5 = plot(residual, lab = "Residual y - (x + f)", color = :black)
plot(p1, p2, p3, p4, p5, ylims = (-4, 4), layout = (5, 1), size = (800, 800))

savefig("./test.svg")