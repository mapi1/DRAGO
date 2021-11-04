using LinearAlgebra
using Plots
using Distributions
using BenchmarkTools
using BandedMatrices

I(1000)[[1,2,5, 200], :]

S = I(5)[[1,2,5], :]

S * S'
diag(S' * S)

function drago(y, λ1, λ2, S = I; Δεmin = 0.001, maxiter = 10, verbose = true)
    # Initialize
    N = length(y)
    x = S * y
    present = [1:N;][diag(S' * S) .== 1]
    Dn = Tridiagonal(ones(N-1), -2ones(N), ones(N-1))


    # Iterate
    Δε = ε = var(y)
    iter = 1
    fRes = similar(y)
    while Δε > Δεmin && iter <= maxiter
        A = Diagonal(λ1 ./ (abs.(x) .+ λ1))
        # B = Matrix(S' * A * S .+ λ2 .* Dn' * Dn) # faster for low N but more Memory gets allocated
        B = BandedMatrix(S' * A * S .+ λ2 .* Dn' * Dn) # less Memory but apparently slower
        fRes = inv(B) * S' * A * S * y
        x = abs.(x) .* (S * (y .- fRes)) ./ (abs.(x) .+ λ1)
        residual = y[present] .- (fRes[present] .+ x)
        εNew = var(residual)
        @show Δε = abs((ε - εNew) / ε) 
        ε = εNew
        verbose && @info "Iteration: $iter with ε = $ε"
        iter += 1
    end
    residual = y[present] .- (fRes[present] .+ x)

    return fRes, x, residual    
end

# Simulate data according to model Sy = Sf + x + w


N = 1000
t = range(0, 50, length = N)
f = -sin.(t) .- cos.(0.5t)

(y, f, x, w, S) = simulateData(f, percGaps = 0.2, maxGapRatio = 0.08)
present = [1:N;][diag(S' * S) .== 1]
(fRes, xRes, residual) = drago(y, 0.6, 50, S, Δεmin = 10e-7)
begin
p1 = plot(f, lab = "Simulated")
p2 = scatter(present, S * y, lab = "Noisy")
p3 = scatter(present, S * y, lab = "", color = :lightgrey, alpha = 0.2)
plot!(fRes, color = :black, linewidth = 2, lab = "Smoothed")
p4 = plot(x[present], color = :red, lab = "True outlier")
plot!(xRes, color = :black, lab = "Estimated outlier") 
p5 = plot(residual, lab = "Residual", color = :black)
plot(p1, p2, p3, p4, p5, ylims = (-3, 3), layout = (5, 1), size = (800, 1000))
end

plot!(xRes)
λ1 = 0.2
λ2 = 0.2
Δεmin = 0.001
Dn = D(N)

plot(xn)

xn = S * y
Δε = 1
iter = 1
while Δε > Δεmin
    A = Diagonal(λ1 ./ (abs.(xn) .+ λ1))
    # B = Matrix(S' * A * S .+ λ2 .* Dn' * Dn) # faster for low N but more Memory gets allocated
    B = BandedMatrix(S' * A * S .+ λ2 .* Dn' * Dn) # less Memory but apparently slower
    fn = inv(B) * S' * A * S * y
    xn = abs.(xn) .* (S * (y .- fn)) ./ (abs.(xn) .+ λ1)
    residual = y[present] .- (fn[present] .+ xn)
    εNew = var(residual)
    Δε = (ε - εNew) / ε 
    ε = εNew
    @info "Iteration: $iter with ε = $ε"
    iter += 1
end


scatter(present, S * y)
plot!(fn)
plot!(present, residual)

plot(S * x)
plot!(xn)

B = D(10000)
@benchmark inv(Matrix(B))
@benchmark inv(BandedMatrix(B))
@benchmark Matrix(B) \ Diagonal(ones(10000))
@benchmark BandedMatrix(B) \ I

B = Matrix(S' * A * S .+ λ2 .* Dn' * Dn) # faster for low N but more Memory gets allocated
B = (S' * A * S .+ λ2 .* Dn' * Dn) # faster for low N but more Memory gets allocated
Bt = Tridiagonal(S' * A * S .+ λ2 .* Dn' * Dn) # less Memory but apprently slower
diag(B, 2)
B .== Bt
scatter(present, S * y)
plot!(fn)
plot(y .- fn)

Bb = BandedMatrix(S' * A * S .+ λ2 .* Dn' * Dn)
@benchmark inv(Matrix(B))
@benchmark inv(Bb)

"""
simulateData(f; percGaps = 0.2, maxGapRatio = 0.3, percOutliers = 0.05, σ_noise = 0.2, outlierStrength = 1) 

Put some gaps and outlier in the smooth signal f. The underlying model takes the form:
Sy = Sf + x + w 

# Keyword Arguments

* percGaps: The percentage of gaps ∈ [0,1]
* maxGapRatio: Maximal allowed ratio between gap length to total gap length ∈ ]0,1] 
* percOutliers: The percentage of outliers ∈ [0,1]
* outlierStrength: Strength of the gaussian outlier distribution
* σ_noise: σ of the additive gaussian noise w

# Returns

(y, f, x, w, S)
"""
function simulateData(f; percGaps = 0.2, maxGapRatio = 0.3, percOutliers = 0.05, σ_noise = 0.2, outlierStrength = 1)
    N = length(f)
    # Gaps
    present = placeGaps(N, percGaps = percGaps, maxGapRatio = maxGapRatio)
    S = I(N)[present, :]
    # Outliers
    x = zeros(N)
    outliers = sample(present, round(Int, percOutliers * N), replace = false) 
    x[outliers] += rand(Normal(0, outlierStrength), length(outliers))
    # Noise
    w = rand(Normal(0, σ_noise), N)
    y = f .+ x .+ w
    return (y, f, x, w, S)
end

function placeGaps(N; percGaps = 0.2, maxGapRatio = 0.3)
    present = collect(1:N)
    toGap = round(Int, percGaps * N)
    maxGap = maxGapRatio * toGap
    while toGap > 0
        gapLen = sample(1:min(maxGap, toGap))
        start = sample(1:(length(present) - gapLen))
        filter!(x -> !(x in start:(start + gapLen - 1)), present)
        toGap = length(present) - (N - round(Int, percGaps * N))
    end
    return present
end