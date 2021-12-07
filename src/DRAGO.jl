module DRAGO

using LinearAlgebra
using Plots
using Distributions
using BandedMatrices

include("utility.jl")

export drago, simulateData, findGaps 

"""
    drago(y, λ1, λ2, S = I; ΔFmin = 0.01, maxiter = 10, verbose = false)

Nonlinear Smoothing of Data with Random Gaps and Outliers (DRAGO). A detailed description of the algorithm can be found in [1]

[1] Parekh, A., et al. "Nonlinear Smoothing of Core Body Temperature Data with Random Gaps and Outliers (DRAGO)." Biomedical Signal Processing. Springer, Cham, 2021. 63-84.
# Arguments

* y:    Signal with gaps and outlier
* λ1:   Influences the sparsity of the estimated outliers `x`. As a reasonable heuristic is λ1 = 3σ, where σ is the standard deviation of the noise. 
* λ2:   Influences the smoothness of the estimate `f`
* S:    A matrix encoding gaps defaulting to `I` if none are present. Otherwise it can be created by starting with `I` and deleting all rows corresponding to a gap. 

# Keyword Arguments

* ΔFmin:    The minimal change in the objective function F that defines convergence.
* maxiter:  Maximum number of iterations
* verbose:  Print some information

# Returns

* fRes:     The smoothed signal
* xRes:     The estimated outliers
* residual: The residual signal y - (f + x)     
"""
function drago(y, λ1, λ2, S = I(length(y)); ΔFmin = 0.01, maxiter = 10, verbose = false)
    # Initialize
    N = length(y)
    xRes = S * y
    present = [1:N;][diag(S' * S) .== 1]
    Dn = Tridiagonal(ones(N-1), -2ones(N), ones(N-1))
    Dn = Dn[2:end-1, 1:end] # make it N-2 x N
    # Dn = BandedMatrix((0 => ones(N-2), 1 => -2ones(N-2), 2 => ones(N-2)), (N-2, N))
    λ2DDn = λ2 .* Dn' * Dn
    
    # Objective function
    F(f, x) = 0.5norm2(S * y .- S * f .- x)^2 + 0.5λ1 * norm1(x) + .5λ2 *  norm2(Dn * f)^2

    # Iterate
    iter = 1
    fRes = similar(y)
    Fold = ΔF = Inf
    while ΔF > ΔFmin && iter <= maxiter
        A = Diagonal(λ1 ./ (abs.(xRes) .+ λ1))
        # B = Matrix(S' * A * S .+ λ2 .* Dn' * Dn) # faster for low N but more Memory gets allocated
        B = BandedMatrix(S' * A * S .+ λ2DDn) # less Memory but apparently slower
        fRes = inv(B) * S' * A * S * y
        xRes = abs.(xRes) .* (S * (y .- fRes)) ./ (abs.(xRes) .+ λ1)
        Fnow = F(fRes, xRes)
        verbose && @info "Iteration: $iter with F = $Fnow"
        ΔF = Fold .- Fnow
        Fold = Fnow 
        iter += 1
    end
    residual = y[present] .- (fRes[present] .+ xRes)
    
    return fRes, xRes, residual    
end
end #module