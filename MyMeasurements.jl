module MyMeasurements
using LinearAlgebra
using Distributions

export my_measurement_logistic, my_measurement_linear

function create_row_orthogonal_matrix(m, n)
    A = rand(Normal(0.0, 1.0/n^0.5), (m, n))
    svd_result = svd(A, full=true)
    u = svd_result.U
    vh = svd_result.Vt
    S = zeros((m,n))
    for i in 1:m
        S[i,i] = 1.0
    end
    A = u * S * vh
    s=ones(m)
    return A, svd_result.U, s, svd_result.V
end

function create_singular_values_exponential(n, δ, κ)
    # これはgeometric setup用
    m = Int(round(n * δ))
    γ = κ^(1.0/(1.0-m))
    s = zeros(m);
    for i in 1:m
        s[i] = γ^(i-1)
    end
    alpha = n / sum(s.^2.0)  # chose the coefficient so that \| A \|_F = n
    s = s .* alpha^0.5;
    return s
end

function create_exponential_matrix(m, n, κ)
    δ = m / n
    A = rand(Normal(0.0, 1.0/n^0.5), (m, n))
    svd_result = svd(A, full=true)
    u = svd_result.U
    vh = svd_result.Vt
    s = create_singular_values_exponential(n, δ, κ)
    S = zeros((m,n))
    for i in 1:m
        S[i,i] = s[i]
    end
    A = u * S * vh

    return A, svd_result.U, s, svd_result.V
end

function create_iidGaussian_matrix(m, n)
    δ = m / n
    A = rand(Normal(0.0, 1.0/n^0.5), (m, n));
    svd_result = svd(A, full=true);
    u = svd_result.U
    vh = svd_result.Vt
    s = svd_result.S

    S = zeros((m,n))
    for i in 1:m
        S[i,i] = s[i]
    end

    return A, svd_result.U, s, svd_result.V
end

function create_signal_gauss_bernoulli(n, ρ)
    x_0 = rand(Binomial(1, ρ), n);
    x_0 = x_0 .* rand(Normal(0.0, 1.0/ρ^0.5), n);
    return x_0
end



"""
logistic回帰用の疑似データの生成. 
"""
function my_measurement_logistic(m, n, ρ, ensemble; κ=10)
    if ensemble == "row_orthogonal"
        A, u, s, v = create_row_orthogonal_matrix(m, n);
    elseif ensemble == "exponential"
        A, u, s, v = create_exponential_matrix(m, n, κ);
    elseif ensemble == "iidgaussian"
        A, u, s, v = create_iidGaussian_matrix(m, n);
    end

    x_0 = rand(Binomial(1, ρ), n)
    x_0 = x_0 .* rand(Normal(0.0, 1.0/ρ^0.5), n)
    z_0 = A * x_0 
    y_array = zeros(Int64, (m, 2))
    y = zeros(m)
    for μ in 1:m
        y[μ] = rand(
            Binomial(1, 1.0 / (1.0 + exp(-z_0[μ])))
        )
        if y[μ] == 1.0
            y_array[μ, 2] = 1
        else
            y_array[μ, 1] = 1
        end
        y[μ] = 2*y[μ] - 1
    end

    return (y=y, y_array=y_array, A=A, x_0=x_0, z_0=z_0, u=u, v=v, s=s)
end

"""
線形回帰用の疑似データの作成
"""
function my_measurement_linear(m, n, ρ, σ, ensemble; κ=10)
    if ensemble == "row_orthogonal"
        A, u, s, v = create_row_orthogonal_matrix(m, n);
    elseif ensemble == "exponential"
        A, u, s, v = create_exponential_matrix(m, n, κ);
    elseif ensemble == "iidGaussian"
        A, u, s, v = create_iidGaussian_matrix(m, n);
    else
        println("not implemented ", ensemble)
    end
    

    x_0 = rand(Binomial(1, ρ), n)
    x_0 = x_0 .* rand(Normal(0.0, 1.0/ρ^0.5), n)
    z_0 = A * x_0 
    y = A * x_0 + rand(Normal(0.0, σ), m);

    return (y=y, A=A, x_0=x_0, z_0=z_0, u=u, v=v, s=s)
end

end