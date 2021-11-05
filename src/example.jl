# Recreate figure 3.3 from the original paper
N = 1000
t = range(0, 50, length = N)
f = -sin.(t) .- cos.(0.5t)
(y, f, x, w, S) = simulateData(f, percGaps = 0.45, maxGapRatio = 0.02, outlierStrength = 1.5)
present = [1:N;][diag(S' * S) .== 1]

(fRes, xRes, residual) = drago(y, 0.6, 500, S, ΔFmin = 10e-2)

# Plotting
begin
    p1 = plot(f, lab = "Simulated")
    p2 = scatter(present, S * y, lab = "Noisy")
    p3 = scatter(present, S * y, lab = "", color = :lightgrey, alpha = 0.2)
    plot!(fRes, color = :black, linewidth = 2, lab = "Smoothed signal with RMSE: $(round(rmse, digits = 2))")
    p4 = plot(x[present], color = :red, lab = "True outlier")
    plot!(xRes, color = :black, lab = "Estimated outlier")
    rmse = norm2(f .- fRes) / norm2(f) 
    p5 = plot(residual, lab = "Residual", color = :black)
    plot(p1, p2, p3, p4, p5, ylims = (-3, 3), layout = (5, 1), size = (800, 800))
end
