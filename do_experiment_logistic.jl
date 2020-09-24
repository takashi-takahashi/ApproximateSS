include("ApproximateSS.jl")
include("MyMeasurements.jl")
using .ApproximateSS
using .MyMeasurements
using Distributions, LinearAlgebra
using GLMNet

m, n = 250, 2000;
ρ = 0.01;

@time y, y_array, A, x_0, z_0, u, v, s =  my_measurement_logistic(m, n, ρ, "row_orthogonal");

@time cv_result = glmnetcv(
    A, y_array, Binomial(),
    intercept=false, standardize=false,
    tol=1.0e-10, maxit=10000, nfolds=10, nlambda=100
)

cv_index = argmin(cv_result.meanloss)
λ = cv_result.lambda[1:cv_index] .* m;
λ_cv = [cv_result.lambda[cv_index]] .* m;
println("$cv_index, $(length(cv_result.lambda))")

### naive experiment ###
function do_SS(A, y_array, n_B, μ_B, λ;  tol=1.0e-10, randomize=false)
    λ₀ = λ
    active_array = zeros((n_B, n, length(λ)));
    first_moment_array = zeros((n_B, n, length(λ)));

    Threads.@threads for b in 1:n_B
        if randomize
            penalty_factor = rand(Binomial(1, 0.5), n) .+ 1.0
        else
            penalty_factor = ones(n)
        end
        λ = λ₀ .* (sum(penalty_factor)/n)

        sample_index = sample(1:m, Int(round(μ_B * m)))
        y_b = y_array[sample_index, :]
        A_b = A[sample_index, :]
        glmnet_result = glmnet(
            A_b, y_b, Binomial(), lambda=λ, 
            intercept=false, standardize=false,
            tol=tol, maxit=100000, 
            penalty_factor=penalty_factor,
        )
        active_array[b, :, :] = (glmnet_result.betas .!= 0.0)
        first_moment_array[b,:, :] = glmnet_result.betas
        if b%10 == 0
            println(b)
        end
    end
    return active_array, first_moment_array
end

n_B = 100
μ_B = 1.0
@time active_array, first_moment_array = do_SS(A, y_array, n_B, μ_B, λ./m, tol=1.0e-12, randomize=true);
Π_experiment = vec(mean(active_array[:, :, cv_index], dims=1));
first_moment_experiment = vec(mean(first_moment_array[:, :, cv_index], dims=1));

### approximate SS ###
@time ss_result = rvamp(A, y, λ_cv, Binomial(), ApproximateSS.Diagonal(), dumping=0.8, t_max=50, debug=false, info=false, pw=0.5, tol=1.0e-6);
@time ss_result_sa = rvamp(A, y, λ_cv, Binomial(), ApproximateSS.DiagonalRestricted(), dumping=0.8, t_max=50, debug=false, info=false, pw=0.5, tol=1.0e-6);

x1_hat_ss = ss_result.x1_hat[1, :];
Π = ss_result.Π[1,:];
x1_hat_ss_sa = ss_result_sa.x1_hat[1, :];
Π_sa = ss_result_sa.Π[1,:];

### visualize results ###
using Plots
p1 = plot(x1_hat_ss, x1_hat_ss)
p1 = plot!(x1_hat_ss, first_moment_experiment, seriestype=:scatter)
p1 = plot!(title="rvamp x1_hat (diagonal)")
p2 = plot(Π, Π, label="")
p2 = plot!(Π, Π_experiment, seriestype=:scatter, label="")
p2 = plot!(title="rvamp Π (diagonal)")
p3 = plot(x1_hat_ss_sa, x1_hat_ss_sa)
p3 = plot!(x1_hat_ss_sa, first_moment_experiment, seriestype=:scatter)
p3 = plot!(title="rvamp x1_hat (diagonal restricted)")
p4 = plot(Π_sa, Π_sa, label="")
p4 = plot!(Π_sa, Π_experiment, seriestype=:scatter, label="")
p4 = plot!(title="rvamp Π (diagonal restricted)")

# plot(p1, p2, size=(1000, 300))
plot(p1, p2, p3, p4, size=(1000, 600))