module ExperimentUtil
using GLMNet
using Distributions

export do_ss

function do_ss(
    A, y, λ, bootstrap_ratio, n_bootstrap_sample;
    randomization=false, 
    tol=1e-12, maxit=1000000, intercept=true, α=1.0)
    λ₀ = λ
    m,n = size(A)
    m_b = Int(round(m * bootstrap_ratio))

    first_moment_array = zeros((n_bootstrap_sample, n, length(λ)))
    selection_array = zeros((n_bootstrap_sample, n, length(λ)))
    intercept_array = zeros((n_bootstrap_sample, length(λ)))

    Threads.@threads for b_index in 1:n_bootstrap_sample
        sample_indexes = sample(1:m, m_b)
        if randomization
            penalty_factor = rand(Binomial(1, 0.5), n) .+ 1.0;
        else
            penalty_factor = ones(n);
        end

        λ = λ₀ .* (sum(penalty_factor)/n)
        A_b = A[sample_indexes, :]
        y_array_b = y[sample_indexes]
        glmnet_b_result = glmnet(
            A_b, y_array_b, Normal(), intercept=intercept, standardize=false,
            tol=tol, maxit=maxit, lambda=λ, penalty_factor=penalty_factor, alpha=α
        )
        first_moment_array[b_index, :, :] = glmnet_b_result.betas
        selection_array[b_index, :, :] = glmnet_b_result.betas.!=0.0
        intercept_array[b_index, :] = glmnet_b_result.a0
        b_index & 10 == 0 && println(b_index)
    end
    # return selection_prob_array, intercept_array
    return selection_array
end


end