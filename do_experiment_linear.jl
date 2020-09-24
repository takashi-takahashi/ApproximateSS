include("./src/ApproximateSS.jl")
include("./src/MyMeasurements.jl")
using .MyMeasurements
using .ApproximateSS
using Distributions, LinearAlgebra
using GLMNet
# using ApproximateSS  # if you installed ApproximateSS...


m, n = 250, 2000;
ρ = 0.1;
σ = 0.2;
intercept = false;
if intercept 
    println("intercept estimation is not implemented for diagonal restricted covariance (self-averaging case)")
    println("(intercept argument will be ignored)")
end
@time y, A, x_0, z_0, u, v, s = my_measurement_linear(m, n, ρ, σ, "row_orthogonal");  # synthetic

@time cv_result = glmnetcv(
    A, y, Normal(), 
    intercept=false, standardize=false, 
    tol=1e-10, maxit=10000, nfolds=10, nlambda=100
)

cv_index = argmin(cv_result.meanloss);
λ = cv_result.lambda[1:cv_index] .* m;
λ_cv = [cv_result.lambda[cv_index]] .* m;
println("$cv_index, $(length(λ))")

### naive experiment ###
function do_SS(A, y, n_B, μ_B, λ; tol=1.0e-10, randomize=false, intercept=false)
    λ₀ = λ

    active_array = zeros((n_B, n, length(λ)));
    first_moment_array = zeros((n_B, n, length(λ)));
    intercept_first_moment_array = zeros((n_B, length(λ)));

    Threads.@threads for b in 1:n_B
        if randomize
            penalty_factor = rand(Binomial(1, 0.5), n) .+ 1.0
        else
            penalty_factor = ones(n)
        end
        λ = λ₀ .* (sum(penalty_factor)/n)

        sample_index = sample(1:m, Int(round(μ_B * m)))
        y_b = y[sample_index]
        A_b = A[sample_index, :]
        glmnet_result = glmnet(
            A_b, y_b, Normal(), lambda=λ, 
            intercept=intercept, standardize=false,
            tol=tol, maxit=100000, 
            penalty_factor=penalty_factor,
        )
        if !intercept
            active_array[b, :, :] = (glmnet_result.betas .!= 0.0)
            first_moment_array[b,:, :] = glmnet_result.betas
        else
            active_array[b, :, :] = (glmnet_result.betas .!= 0.0)
            first_moment_array[b,:, :] = glmnet_result.betas
            intercept_first_moment_array[b,:] = glmnet_result.a0
        end
        if b%10 == 0
            println(b)
        end
    end
    return active_array, first_moment_array, intercept_first_moment_array
end
n_B = 10000
μ_B = 1.0

@time active_array, first_moment_array, intercept_first_moment_array = do_SS(
        A, y, n_B, μ_B, λ./m, tol=1.0e-12, randomize=true, intercept=intercept
    );
Π_experiment = vec(mean(active_array[:, :, cv_index], dims=1));
first_moment_experiment = vec(mean(first_moment_array[:, :, cv_index], dims=1));
intercept_experiment = mean(intercept_first_moment_array[:, cv_index])

### approximate SS ###
@time ss_result = rvamp(
    A, y, λ_cv, Normal(), ApproximateSS.Diagonal(), 
    dumping=0.8, t_max=50, debug=false, info=false, pw=0.5, tol=1.0e-10, intercept=intercept
);

@time ss_result_sa = rvamp(
        A, y, λ_cv, Normal(), ApproximateSS.DiagonalRestricted(),
        dumping=0.8, t_max=50, debug=false, info=false, pw=0.5, tol=1.0e-6, intercept=intercept
    );

x1_hat_ss = zeros(n);
Π = zeros(n);
x1_hat_ss_sa = zeros(n);
Π_sa = zeros(n);
if intercept
    x1_hat_ss = ss_result.x1_hat[1, 2:n+1];
    Π = ss_result.Π[1,2:n+1];
    x1_hat_ss_sa = ss_result_sa.x1_hat[1, :];
    Π_sa = ss_result_sa.Π[1,:];
    println("intercept(experiment): $(intercept_experiment)")
    println("intercept(rvamp (diagonal covariance)): $(ss_result.x1_hat[1])")
else
    x1_hat_ss = ss_result.x1_hat[1, :];
    Π = ss_result_sa.Π[1,:];
    x1_hat_ss_sa = ss_result_sa.x1_hat[1, :];
    Π_sa = ss_result_sa.Π[1,:];        
end




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

plot(p1, p2, p3, p4, size=(1000, 600))