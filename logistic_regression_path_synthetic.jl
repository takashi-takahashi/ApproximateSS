include("./src/ApproximateSS.jl")
include("./src/MyMeasurements.jl")
using .ApproximateSS
# using ApproximateSS
using .MyMeasurements
using Distributions, LinearAlgebra
using GLMNet

m, n = 250, 2000;
ρ = 0.01;
intercept = false;
if intercept 
    println("intercept estimation is not implemented for diagonal restricted covariance (self-averaging case)")
    println("(intercept argument will be ignored)")
end

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
function do_SS(A, y_array, n_B, μ_B, λ;  tol=1.0e-10, randomize=false, intercept=false)
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
        y_b = y_array[sample_index, :]
        A_b = A[sample_index, :]
        glmnet_result = glmnet(
            A_b, y_b, Binomial(), lambda=λ, 
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
        A, y_array, n_B, μ_B, λ./m, tol=1.0e-12, randomize=true, intercept=intercept
    );
Π_experiment = mean(active_array, dims=1)[1,:,:]';
first_moment_experiment = mean(first_moment_array, dims=1)[1,:,:]';
intercept_experiment = mean(intercept_first_moment_array, dims=1);
    

### approximate SS ###
@time ss_result = rvamp(
        A, y, λ, Binomial(), ApproximateSS.Diagonal(),
        dumping=0.8, t_max=50, debug=false, info=true, pw=0.5, tol=1.0e-6, intercept=intercept
    );
@time ss_result_sa = rvamp(
        A, y, λ, Binomial(), ApproximateSS.DiagonalRestricted(),
        dumping=0.8, t_max=50, debug=false, info=true, pw=0.5, tol=1.0e-6, intercept=intercept
    );


x1_hat_ss = zeros((length(λ), n));
Π = zeros((length(λ), n));
x1_hat_ss_sa = zeros((length(λ), n));
Π_sa = zeros((length(λ), n));

if intercept
    x1_hat_ss = ss_result.x1_hat[:, 2:n+1];
    Π = ss_result.Π[:, 2:n+1];
    x1_hat_ss_sa = ss_result_sa.x1_hat[:, :];
    Π_sa = ss_result_sa.Π[:,:];
    println("intercept(experiment): $(intercept_experiment)")
    println("intercept(rvamp (diagonal covariance)): $(ss_result.x1_hat[1])")
else
    x1_hat_ss = ss_result.x1_hat[:, :];
    Π = ss_result.Π[:,:];
    x1_hat_ss_sa = ss_result_sa.x1_hat[:, :];
    Π_sa = ss_result_sa.Π[:,:];        
end

# 実験のほうのstability(cv optimalのところ)を降順にソートして、stabilityが大きいやつだけpathを描く（全部描くと何もわからん）
sorted_index = [x[2] for x in sort([(x, y) for (x, y) in zip(Π_experiment[length(λ),:], 1:n)], by=x->x[1],rev=true)[:, 1]];  # 降順

n_path = 10  # pathの本数


### visualize results ###
using Plots
using LaTeXStrings

p1 = plot(λ, x1_hat_ss[:, sorted_index[1:n_path]], xscale=:log10, label="", color=:blue)
p1 = plot!(λ, first_moment_experiment[:, sorted_index[1:n_path]], color=:red, label="", linestyle=:dash)
p1 = plot!(title="x1_hat (diagonal, blue:rvamp, red:naive)", xlabel=L"\lambda")
p2 = plot(λ, Π[:, sorted_index[1:n_path]], label="", color=:blue)
p2 = plot!(λ, Π_experiment[:, sorted_index[1:n_path]], label="", color=:red, xscale=:log10, linestyle=:dash)
p2 = plot!(title="Π (diagonal, blue:rvamp, red:naive)", xlabel=L"\lambda")

p3 = plot(λ, x1_hat_ss_sa[:, sorted_index[1:n_path]], xscale=:log10, label="", color=:blue)
p3 = plot!(λ, first_moment_experiment[:, sorted_index[1:n_path]], color=:red, label="", linestyle=:dash)
p3 = plot!(title="x1_hat (diagonal restricted, blue:rvamp, red:naive)", xlabel=L"\lambda")
p4 = plot(λ, Π_sa[:, sorted_index[1:n_path]], label="", color=:blue)
p4 = plot!(λ, Π_experiment[:, sorted_index[1:n_path]], label="", color=:red, xscale=:log10, linestyle=:dash)
p4 = plot!(title="Π (diagonal restricted, blue:rvamp, red:naive)", xlabel=L"\lambda")

plot(p1, p2, p3, p4, size=(1000, 600))