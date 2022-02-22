"""
find_gaps(time, Δt = median(diff(time)); radius = 0.1Δt)

Find gaps in a time vector and encode them into a matrix S. S' * y results in the original signal with zero filling.

# Arguments

* time:     A vector that encodes the time (eg. seconds, samples, ...)
* Δt:       The expected distance between two measurements, defaults to median(diff(time)) 
* radius:   If the signal is not non-uniformly sampled, the radius can be used to search in a broader area around the expected sample. 
""" 
function find_gaps(time, Δt = median(diff(time)); radius = 0.1Δt)
    present = Bool[]
    diffTime = diff(time)
    currentGapLen = 1
    push!(present, true)
    for difft in diffTime
        while currentGapLen * Δt + radius < difft
            push!(present, false)
            currentGapLen += 1 
        end
        push!(present, true)
        currentGapLen = 1 
    end
    S = I(length(present))[present, :]
    return S    
end

"""
simulate_data(f; percentage_gaps = 0.2, max_gap_ratio = 0.3, percentage_outliers = 0.05, σ_noise = 0.2, outlier_strength = 1) 

Put some gaps and outlier in the smooth signal f. The underlying model takes the form:
Sy = Sf + x + w 

# Keyword Arguments

* percentage_gaps: The percentage of gaps ∈ [0,1]
* max_gap_ratio: Maximal allowed ratio between gap length to total gap length ∈ ]0,1] 
* percentage_outliers: The percentage of outliers ∈ [0,1]
* outlier_strength: Strength of the gaussian outlier distribution
* σ_noise: σ of the additive gaussian noise w

# Returns

(y, x, w, S)
"""
function simulate_data(f; percentage_gaps = 0.4, max_gap_ratio = 0.1, percentage_outliers = 0.05, σ_noise = 0.2, outlier_strength = 1)
    N = length(f)
    # Gaps
    present = place_gaps(N, percentage_gaps = percentage_gaps, max_gap_ratio = max_gap_ratio)
    S = I(N)[present, :]
    # Outliers
    x = zeros(N)
    outliers = sample(present, round(Int, percentage_outliers * N), replace = false) 
    x[outliers] += rand(Normal(0, outlier_strength), length(outliers))
    # Noise
    w = rand(Normal(0, σ_noise), N)
    y = f .+ x .+ w
    return (y, x, w, S)
end

"""
Get the indices where data is present from the encoding matrix S 
"""
get_present(S) = [1:size(S,2);][diag(S' * S) .== 1]


# Distribute the gaps onto the signal, return which samples are present
function place_gaps(N; percentage_gaps = 0.2, max_gap_ratio = 0.3)
    present = collect(1:N)
    total_gaps = round(Int, percentage_gaps * N)
    max_gap = max_gap_ratio * total_gaps
    while total_gaps > 0
        gap_len = sample(1:min(max_gap, total_gaps))
        start = sample(1:(length(present) - gap_len))
        filter!(x -> !(x in start:(start + gap_len - 1)), present)
        total_gaps = length(present) - (N - round(Int, percentage_gaps * N))
    end
    return present
end

# L1 norm
norm1(x) = sum(abs, x)

# L2 norm
norm2(x) = sqrt(sum(abs2, x))
