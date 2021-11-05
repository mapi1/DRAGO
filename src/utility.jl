
# N = 10000
# B = Tridiagonal(ones(N-1), -2ones(N), ones(N-1))
# @benchmark inv(Matrix(B))
# @benchmark inv(BandedMatrix(B))
# @benchmark Matrix(B) \ Diagonal(ones(10000))
# @benchmark BandedMatrix(B) \ I

"""
simulateData(f; percGaps = 0.2, maxGapRatio = 0.3, percOutliers = 0.05, σ_noise = 0.2, outlierStrength = 1) 

Put some gaps and outlier in the smooth signal f. The underlying model takes the form:
Sy = Sf + x + w 

# Keyword Arguments

* percGaps: The percentage of gaps ∈ [0,1]
* maxGapRatio: Maximal allowed ratio between gap length to total gap length ∈ ]0,1] 
* percOutliers: The percentage of outliers ∈ [0,1]
* outlierStrength: Strength of the gaussian outlier distribution
* σNoise: σ of the additive gaussian noise w

# Returns

(y, f, x, w, S)
"""
function simulateData(f; percGaps = 0.2, maxGapRatio = 0.3, percOutliers = 0.05, σNoise = 0.2, outlierStrength = 1)
    N = length(f)
    # Gaps
    present = placeGaps(N, percGaps = percGaps, maxGapRatio = maxGapRatio)
    S = I(N)[present, :]
    # Outliers
    x = zeros(N)
    outliers = sample(present, round(Int, percOutliers * N), replace = false) 
    x[outliers] += rand(Normal(0, outlierStrength), length(outliers))
    # Noise
    w = rand(Normal(0, σNoise), N)
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

# L1 norm
norm1(x) = sum(abs, x)

# L2 norm
norm2(x) = sqrt(sum(abs2, x))
