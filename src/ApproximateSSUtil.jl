module ApproximateSSUtil

using LinearAlgebra
using Distributions
using FastGaussQuadrature
using SpecialFunctions
using QuadGK

export normal_cdf, normal_pdf, normal_sf, heaviside, my_newton_solver, my_newton_solver_b,
    poisson_expect_sc, dumping, poisson_const, exponential_eigen_dist, average_exponential_eigen,
    gauss_hermite_integral, woodbury


function normal_pdf(x::Float64)
    pdf(Normal(0.0, 1.0), x)
end

function normal_cdf(x::Float64)
    cdf(Normal(0.0, 1.0), x)
end

function normal_sf(x::Float64)
    1.0 .- cdf(Normal(0.0, 1.0), x)
end

function heaviside(h::Float64, threshold::Float64)
    ifelse(abs(h) - threshold >= 0.0, 1.0, 0.0)
end

"""
ブートストラップしていない場合の、logistic回帰用のNewton法solver
"""
function my_newton_solver(q1z_hat::Float64, h1z::Float64, y::Float64,
    x0=0.0::Float64, tol=1e-9::Float64, max_iter=100::Int64)::Float64
    x_t = x0
    for t in 1:max_iter
        pre_x = x_t
        numerator = (q1z_hat * x_t - h1z - y / (1.0 + exp(y * x_t)))
        denominator = q1z_hat + 0.25 / cosh(0.5 * x_t)^2.0
        x_t -= numerator / denominator
        (abs(x_t - pre_x)<tol) && return x_t
    end
    return x_t
end

"""
ブートストラップしている場合の、logistic回帰用ニュートン法solver
"""
function my_newton_solver_b(q1z_hat::Float64, h1z::Float64, y::Float64, c::Int64,
    x0=0.0::Float64, tol=1e-9::Float64, max_iter=100::Int64)::Float64
    x_t = x0
    for t in 1:max_iter
        pre_x = x_t
        numerator = (q1z_hat * x_t - h1z - y * c / (1.0 + exp(y * x_t)))
        denominator = q1z_hat + c * 0.25 / cosh(0.5 * x_t)^2.0
        x_t -= numerator / denominator
        (abs(x_t - pre_x)<tol) && return x_t
    end
    return x_t
end

function poisson_expect_sc(func, μ, max_c=100)::Float64
    value = func(0) * pdf(Poisson(μ), 0)
    for c in 1:max_c
        value += func(c) * pdf(Poisson(μ), c)
    end
    value
end

"""
ブートストラップなし
"""
function poisson_const(func, μ, max_c)::Float64
    return func(1)
end


function dumping(x, y, η)
    return η * x + (1.0 - η) * y
end


function exponential_eigen_dist(x, δ, κ)
    γ = log(κ);
    return 1.0/(2.0 * γ * x)
end

function average_exponential_eigen(f, δ, κ)
    γ = log(κ);
    a_kappa = sqrt(
        (2.0/δ) * (γ / (1.0 - exp(-2γ)))
    )
    λ_min = (a_kappa/κ)^2.0  # minimum eigen value
    λ_max = a_kappa^2.0  # maximum eigen value
    ρ(x) = exponential_eigen_dist(x, δ, κ);
    return quadgk(x->f(x)*ρ(x), λ_min, λ_max)[1]
end

ξ, w_ξ = gausshermite(100);
function gauss_hermite_integral(f)
    return sum(f.(ξ) .* w_ξ)/π^0.5
end

"""
woodbury公式
"""
function woodbury(A::Array{Float64, 2},q2x_hat::Array{Float64, 1}, 
    q2z_hat::Array{Float64,1})::Tuple{Array{Float64, 2}, Array{Float64,2}}
    m, n = size(A)
    q2x_hat_inv = 1.0 ./ q2x_hat
    q2z_hat_inv = 1.0 ./ q2z_hat
    temp = q2x_hat_inv' .* A
    temp_A = temp * A'
    temp_inv = inv(diagm(q2z_hat_inv) + A * temp')
    B = -temp' * temp_inv * temp

    @simd for i in 1:n
        B[i,i] += q2x_hat_inv[i]
    end
    C = temp_A - temp_A * temp_inv * temp_A 
    return B, C
end


end