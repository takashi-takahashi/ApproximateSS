module ApproximateSS  # stard module

include("ApproximateSSUtil.jl")
using Distributions, LinearAlgebra
using .ApproximateSSUtil

export Diagonal, DiagonalRestricted  # covariance type
export RVAMPDiaginal, RVAMPDiagonalRestricted
export VAMPDiagonal, RVAMPDiagonalRestricted
export rvamp  # approximate stability selection algorithm
export vamp  # 普通のVAMP

"""
covariance class
"""
abstract type CovarianceType end

"""
diagonal covariance
"""
struct Diagonal <: CovarianceType
end

"""
diagonal restricted covariance
(a.k.a. self-averaging)
"""
struct DiagonalRestricted <: CovarianceType
end

"""
rvamp result (diagonal)
複数の正則化パラメータに対して結果を保持することを想定している。
"""
struct RVAMPDiagonal
    q1x_hat::Array{Float64, 2}  # Onsager coefficient
    v1x_hat::Array{Float64, 2}  # variance conjugate 
    h1x::Array{Float64, 2}  # local field 
    Π::Array{Float64, 2}  # stability
    x1_hat::Array{Float64, 2}  # first moment
end


"""
rvamp result (diagonal restricted)
複数の正則化パラメータに対して結果を保持することを想定している。
"""
struct RVAMPDiagonalRestricted
    q1x_hat::Array{Float64,1}  # Onsager coefficient
    v1x_hat::Array{Float64,1}  # variance conjugate
    h1x::Array{Float64, 2}  # local field 
    Π::Array{Float64, 2}  # stability　
    x1_hat::Array{Float64, 2}  # first moment
end

"""
vamp result (diagonal)
複数の正則化パラメータに対して結果を保持することを想定している。
"""
struct VAMPDiagonal
    q1x_hat::Array{Float64, 2}  # Onsager coefficient
    h1x::Array{Float64, 2}  # local field 
    x1_hat::Array{Float64, 2}  # estimator
end

"""
vamp result (diagonal restricted)
複数の正則化パラメータに対して結果を保持することを想定している。
"""
struct VAMPDiagonalRestricted
    q1x_hat::Array{Float64, 1}  # Onsager coefficient
    h1x::Array{Float64, 2}  # local field 
    x1_hat::Array{Float64, 2}  # estimator
end


"""
replicated vector AMP (diagonal covariance)
linear regression
"""
function rvamp(
        A::Array{Float64, 2}, y::Array{Float64,1}, λ::Array{Float64, 1}, family::Normal{Float64}, cov::Diagonal;
        intercept=false, dumping=0.85, t_max = 30, n_points=15, tol=1e-8, 
        bootstrap_ratio=1.0, pw=0.0, w=2.0,
        clamp_min=1.0e-9, clamp_max=1.0e9, info=true, debug=false
    )::RVAMPDiagonal
    println("approximate stability selection for linear regression")
    println("(diagonal covariance)")

    # preparation
    μ = bootstrap_ratio
    m, n = size(A);  # system size
    c_max = Int(round(μ * 100))
    # c_array = convert(Array{Float64}, 0:c_max);
    # poisson_weight = [pdf(Poisson(μ), c) for c in 0:c_max];
    

    q1x_hat_array = zeros((length(λ), n));
    v1x_hat_array = zeros((length(λ), n));
    h1x_array = zeros((length(λ), n));
    x1_hat_array = zeros((length(λ), n));
    Π_array = zeros((length(λ), n));
    
    h1x = rand(Normal(0.0, 1.0), n);
    q1x_hat = ones(n);
    v1x_hat = ones(n);
    h2x = zeros(n);
    q2x_hat = zeros(n);
    v2x_hat = zeros(n);

    x1_hat = zeros(n);
    chi1x = zeros(n);
    v1x = zeros(n);
    x2_hat = zeros(n);
    chi2x = zeros(n);
    v2x = zeros(n);

    h1z = rand(Normal(0.0, 1.0), m);
    q1z_hat = ones(m);
    v1z_hat = ones(m);
    h2z = zeros(m);
    q2z_hat = zeros(m);
    v2z_hat = zeros(m);

    z1_hat = zeros(m);
    chi1z = zeros(m);
    v1z = zeros(m);
    z2_hat = zeros(m);
    chi2z = zeros(m);
    v2z = zeros(m);

    B = zeros((m, m));
    C = zeros((m, m));
    d = zeros(n);
    BC = zeros((m, m));
    temp = zeros((m, m));
    temp2 = zeros((m, m));
    temp3 = zeros((m, m));
    h_tilde = zeros(n);
    D = zeros((m, m));
    F = zeros((m, m));
    q2x_hat_inv = zeros(n);
    q2z_hat_inv = zeros(m);

    η_p = zeros(n);
    η_m = zeros(n);
    η_p2 = zeros(n);
    η_m2 = zeros(n);


    for (λ_index, γ) in enumerate(λ)
        if info
            println("$λ_index, $γ")
        end
        
        for t in 1:t_max
            # factorized part
            if debug
                println("factorized part")
            end
            ## x
            if debug
                println("(x)")
            end
            η_p .= (γ .- h1x) ./ v1x_hat.^0.5
            η_m .= (-1.0).*(γ .+ h1x) ./ v1x_hat.^0.5

            η_p2 .= (w .* γ .- h1x) ./ v1x_hat.^0.5
            η_m2 .= (-1.0) .* (w .* γ .+ h1x) ./ v1x_hat.^0.5
            
            x1_hat .= (1.0 .- pw) .* (
                (h1x .- γ) .* normal_sf.(η_p) .+ v1x_hat.^0.5 .* normal_pdf.(η_p)
                .+
                (h1x .+ γ) .* normal_cdf.(η_m) .- v1x_hat.^0.5 .* normal_pdf.(η_m)
            ) ./ q1x_hat .+ pw .*(
                (h1x .- w .* γ) .* normal_sf.(η_p2) .+ v1x_hat.^0.5 .* normal_pdf.(η_p2)
                .+
                (h1x .+ w .* γ) .* normal_cdf.(η_m2) .- v1x_hat.^0.5 .* normal_pdf.(η_m2)
            ) ./ q1x_hat

            
            chi1x .= clamp.(
                (1.0 .- pw) .* (
                    normal_sf.(η_p) .+ normal_cdf.(η_m)
                ) ./ q1x_hat .+ pw .* (
                    normal_sf.(η_p2) .+ normal_cdf.(η_m2)
                ) ./ q1x_hat,
                clamp_min, clamp_max
            )

            v1x .= clamp.(
                (1.0 .- pw) .* (
                    ((h1x .- γ).^2.0 .+ v1x_hat).*normal_sf.(η_p) .+ (2.0.*(h1x .- γ).* v1x_hat.^0.5 .+ η_p .* v1x_hat ).*normal_pdf.(η_p)
                    .+
                    ((h1x .+ γ).^2.0 .+ v1x_hat).*normal_cdf.(η_m) .- (2.0.*(h1x .+ γ).* v1x_hat.^0.5 .+ η_m .* v1x_hat ).*normal_pdf.(η_m)
                )./q1x_hat.^2.0 .+ pw .* (
                    ((h1x .- w .* γ).^2.0 .+ v1x_hat).*normal_sf.(η_p2) .+ (2.0.*(h1x .- w .* γ).* v1x_hat.^0.5 .+ η_p2 .* v1x_hat ).*normal_pdf.(η_p2)
                    .+
                    ((h1x .+ w .* γ).^2.0 .+ v1x_hat).*normal_cdf.(η_m2) .- (2.0.*(h1x .+ w .* γ).* v1x_hat.^0.5 .+ η_m2 .* v1x_hat ).*normal_pdf.(η_m2)
                ) ./ q1x_hat .^ 2.0
                .-
                x1_hat.^2.0,
                clamp_min, clamp_max
            )

            ## z
            if debug
                println("(z)")
            end
            # z1_hat .= (h1z .+ y) ./ (1.0 .+ q1z_hat)
            # chi1z .= clamp.(
            #     1.0 ./ (1.0 .+ q1z_hat),
            #     clamp_min, clamp_max
            # )
            # v1z .= clamp.(
                # ((h1z .+ y)./(1.0 .+ q1z_hat)).^2.0 
                #     .+ 
                #     v1z_hat ./　(1.0 .+ q1z_hat).^2.0 
            #     clamp_min, clamp_max
            # )
            Threads.@threads for j in 1:m
                if info && j==1
                    println("z1_hat")
                end
                z1_hat[j] = poisson_expect_sc(
                    c-> (h1z[j] + c*y[j]) / (c + q1z_hat[j]),
                    μ, c_max
                )
                if info && j==1
                    println("chi1z")
                end
                chi1z[j] = clamp(
                        poisson_expect_sc(
                        c->1.0/(c + q1z_hat[j]),
                        μ, c_max
                    ),
                    clamp_min, clamp_max
                )
                if info && j==1
                    println("v1z")
                end
                v1z[j] = clamp(
                    poisson_expect_sc(
                            c->(
                                ((h1z[j] + c*y[j])./(c + q1z_hat[j]))^2.0 
                                + 
                                v1z_hat[j] /　(c + q1z_hat[j])^2.0 
                            ), μ, c_max
                    ) - z1_hat[j]^2.0,
                    clamp_min, clamp_max
                )
            end

            # message passing (factorized -> Gaussian)
            if debug
                println("message passing (factorized -> Gaussian)")
            end
            h2x .= x1_hat ./ chi1x .- h1x
            q2x_hat .= clamp.(
                1.0./chi1x .- q1x_hat,
                clamp_min, clamp_max
            )
            v2x_hat .= clamp.(
                v1x ./ chi1x.^2.0 .- v1x_hat,
                clamp_min, clamp_max
            )
            h2z .= z1_hat ./ chi1z .- h1z
            q2z_hat .= clamp.(
                1.0./chi1z .- q1z_hat,
                clamp_min, clamp_max
            )
            v2z_hat .= clamp.(
                v1z ./ chi1z.^2.0 .- v1z_hat,
                clamp_min, clamp_max
            )
            
            

            # Gaussian part
            if debug
                println("Gaussian part")
            end
            # ---- naive imprementation ----
            # X .= inv(diagm(q2x_hat) + A' * diagm(q2z_hat) * A)
            # Y .= diagm(v2x_hat) + A'*diagm(v2z_hat)*A
            # ## x
            # if debug
            #     println("(x)")
            # end
            # x2_hat .= X * (h2x .+ A'*h2z)
            # chi2x .= clamp.(
            #     diag(X),
            #     clamp_min, clamp_max
            # )
            # v2x .= clamp.(
            #     diag(X * Y * X),
            #     clamp_min, clamp_max
            # )
            # ## z
            # if debug
            #     println("(y)")
            # end
            # z2_hat .= A * x2_hat
            # chi2z .= clamp.(
            #     diag(A*X*A'),
            #     clamp_min, clamp_max
            # )
            # v2z .= clamp.(
            #     diag(A*X*Y*X*A'),
            #     clamp_min, clamp_max
            # )
            q2x_hat_inv .= 1.0 ./ q2x_hat
            q2z_hat_inv .= 1.0 ./ q2z_hat        
            B .= (q2x_hat_inv' .* A) * A'  # m x m
            C .= inv(LinearAlgebra.Diagonal(q2z_hat_inv) .+ B)  # m x m
            D .= ((v2x_hat.*q2x_hat_inv.^2.0)'.*A) * A';  # m x m
            F .= (v2z_hat' .* B) * B';  # m x m

            ## d .= vec(sum(A' .* (C*A)' , dims=2))  # n
            d .= dot.(eachrow(A'), eachcol(C*A))
            h_tilde .= (h2x .+ A' * h2z) .* q2x_hat_inv  # n
            BC .= B*C  # m x m
            temp .= C * D * C;  # m x m
            temp2 .= LinearAlgebra.Diagonal(v2z_hat) * BC;  # m * m
            temp3 .= BC'* (v2z_hat .* BC);  # m x m
            

            ## x
            x2_hat .= h_tilde .- q2x_hat_inv .* (A' * (C*(A*h_tilde)));
            chi2x .= q2x_hat_inv .- d .* q2x_hat_inv.^2.0;

            v2x .= (
                v2x_hat
                .- 2.0 .* v2x_hat .* q2x_hat_inv .* d
                ##.+ vec(sum(A' .* (temp * A)', dims=2))  # ここまで初項
                .+ dot.(eachrow(A'), eachcol(temp * A))  # ここまで初項
                # ここから第二項
                # .+ vec(sum(A' .* (v2z_hat .* A)', dims=2))
                # .- 2.0 .* vec(sum(A' .* (temp2 * A)', dims=2))
                # .+ vec(sum(A'.*(temp3*A)', dims=2))
                .+ dot.(eachrow(A'), eachcol(v2z_hat .* A))
                .- 2.0 .* dot.(eachrow(A'), eachcol(temp2 * A))
                .+ dot.(eachrow(A'), eachcol(temp3 * A))
            ) .* q2x_hat_inv.^2.0

            ## z
            z2_hat .= A * x2_hat
            chi2z .= diag(B) .- diag(B*C*B)
            v2z .= diag(
                D .- 2.0 .* D*BC' .+ BC*D*BC'
            ) .+ diag(
                F .- 2.0 .* F * BC' .+ BC*F*BC'
            )


            # message passing (Gaussian -> factorized)
            h1x .= dumping .* (x2_hat ./ chi2x .- h2x) .+ (1.0 .- dumping) .* h1x
            q1x_hat .= dumping .* (1.0 ./ chi2x .- q2x_hat) .+ (1.0 .- dumping) .* q1x_hat
            v1x_hat .= dumping .* (v2x ./ chi2x.^2.0 .- v2x_hat) .+ (1.0 .- dumping) .* v1x_hat

            ## z
            h1z .= dumping .* (z2_hat ./ chi2z .- h2z) .+ (1.0 .- dumping) .* h1z
            q1z_hat .= dumping .* (1.0 ./ chi2z .- q2z_hat) .+ (1.0 .- dumping) .* q1z_hat
            v1z_hat .= dumping .* (v2z ./ chi2z.^2.0 .- v2z_hat) .+ (1.0 .- dumping) .* v1z_hat

            diff_x = mean((x1_hat - x2_hat).^2.0)
            diff_z = mean((z1_hat - z2_hat).^2.0)
            diff = maximum([diff_x, diff_z])

            if info
                println("t = $t, diff = $diff, μ=$μ, pw=$pw, w=$w, γ=$γ")
                println("\t $(mean(chi1x)),\t $(mean(v1x)),\t  $(mean(q1x_hat)),\t $(mean(v1x_hat))")
                println("\t $(mean(chi1z)),\t $(mean(v1z)),\t  $(mean(q1z_hat)),\t $(mean(v1z_hat))")
                println("\t $(minimum(chi1x)),\t $(minimum(v1x)),\t  $(minimum(q1x_hat)),\t $(minimum(v1x_hat))")
                println("\t $(minimum(chi1z)),\t $(minimum(v1z)),\t  $(minimum(q1z_hat)),\t $(minimum(v1z_hat))")
            end
            q1x_hat .= clamp.(q1x_hat, clamp_min, clamp_max)
            v1x_hat .= clamp.(v1x_hat, clamp_min, clamp_max)
            q1z_hat .= clamp.(q1z_hat, clamp_min, clamp_max)
            v1z_hat .= clamp.(v1z_hat, clamp_min, clamp_max)
            if diff < tol
                break
            end
        end

        q1x_hat_array[λ_index, :] = q1x_hat
        v1x_hat_array[λ_index, :] = v1x_hat
        h1x_array[λ_index, :] = h1x
        Π_array[λ_index, :] = chi1x .* q1x_hat
        x1_hat_array[λ_index, :] = x1_hat

        if info 
            println()
        end
    end

    result = RVAMPDiagonal(q1x_hat_array, v1x_hat_array, h1x_array, Π_array, x1_hat_array);
    return result
end

"""
replicated vector AMP (diagonal restricted covariance)
linear regression
"""
function rvamp(
        A::Array{Float64, 2}, y::Array{Float64,1}, λ::Array{Float64, 1}, family::Normal{Float64}, cov::DiagonalRestricted;
        intercept=false, dumping=0.85, t_max=30, n_points=15, tol=1e-8, 
        bootstrap_ratio=1.0, pw=0.0, w=2.0,
        clamp_min=1.0e-9, clamp_max=1.0e9, info=true, debug=false
    )::RVAMPDiagonalRestricted
    println("approximate stability selection for linear regression")
    println("(diagonal restricted covariance)")

    # preparation
    μ = bootstrap_ratio
    m, n = size(A);  # system size
    c_max = Int(round(μ * 100))
    # c_array = convert(Array{Float64}, 0:c_max);
    # poisson_weight = [pdf(Poisson(μ), c) for c in 0:c_max];
    

    q1x_hat_array = zeros((length(λ)));
    v1x_hat_array = zeros((length(λ)));
    h1x_array = zeros((length(λ), n));
    x1_hat_array = zeros((length(λ), n));
    Π_array = zeros((length(λ), n));
    
    h1x = rand(Normal(0.0, 1.0), n);
    q1x_hat = 1.0;
    v1x_hat = 1.0;
    h2x = zeros(n);
    q2x_hat = 1.0;
    v2x_hat = 1.0;

    x1_hat = zeros(n);
    chi1x = 1.0;
    v1x = 1.0;
    x2_hat = zeros(n);
    chi2x = 1.0;
    v2x = 1.0;

    h1z = rand(Normal(0.0, 1.0), m);
    q1z_hat = 1.0;
    v1z_hat = 1.0;
    h2z = zeros(m);
    q2z_hat = 1.0;
    v2z_hat = 1.0;

    z1_hat = zeros(m);
    chi1z = 1.0;
    chi1z_temp = zeros(m);
    v1z = 1.0;
    v1z_temp = zeros(m);
    z2_hat = zeros(m);
    chi2z = 1.0;
    v2z = 1.0;

    svd_result = svd(A);
    u = svd_result.U;
    v = svd_result.V;
    s_m = svd_result.S;
    s_n = zeros(n);
    s_n[1:m] = s_m;
    temp = zeros(n);
    
    η_p = zeros(n);
    η_m = zeros(n);
    η_p2 = zeros(n);
    η_m2 = zeros(n);


    for (λ_index, γ) in enumerate(λ)
        if info
            println("$λ_index, $γ")
        end
        
        for t in 1:t_max
            # factorized part
            if debug
                println("factorized part")
            end
            ## x
            if debug
                println("(x)")
            end
            η_p .= (γ .- h1x) ./ v1x_hat.^0.5
            η_m .= (-1.0).*(γ .+ h1x) ./ v1x_hat.^0.5
            η_p2 .= (w .* γ .- h1x) ./ v1x_hat.^0.5
            η_m2 .= (-1.0) .* (w .* γ .+ h1x) ./ v1x_hat.^0.5
            
            x1_hat .= (1.0 .- pw) .* (
                (h1x .- γ) .* normal_sf.(η_p) .+ v1x_hat.^0.5 .* normal_pdf.(η_p)
                .+
                (h1x .+ γ) .* normal_cdf.(η_m) .- v1x_hat.^0.5 .* normal_pdf.(η_m)
            ) ./ q1x_hat .+ pw .*(
                (h1x .- w .* γ) .* normal_sf.(η_p2) .+ v1x_hat.^0.5 .* normal_pdf.(η_p2)
                .+
                (h1x .+ w .* γ) .* normal_cdf.(η_m2) .- v1x_hat.^0.5 .* normal_pdf.(η_m2)
            ) ./ q1x_hat

            
            chi1x = clamp.(
                mean(
                    (1.0 .- pw) .* (
                        normal_sf.(η_p) .+ normal_cdf.(η_m)
                    ) ./ q1x_hat .+ pw .* (
                        normal_sf.(η_p2) .+ normal_cdf.(η_m2)
                    ) ./ q1x_hat
                ),
                clamp_min, clamp_max
            )

            v1x = clamp.(
                    mean(
                        (1.0 .- pw) .* (
                            ((h1x .- γ).^2.0 .+ v1x_hat).*normal_sf.(η_p) .+ (2.0.*(h1x .- γ).* v1x_hat.^0.5 .+ η_p .* v1x_hat ).*normal_pdf.(η_p)
                            .+
                            ((h1x .+ γ).^2.0 .+ v1x_hat).*normal_cdf.(η_m) .- (2.0.*(h1x .+ γ).* v1x_hat.^0.5 .+ η_m .* v1x_hat ).*normal_pdf.(η_m)
                        )./q1x_hat.^2.0 .+ pw .* (
                            ((h1x .- w .* γ).^2.0 .+ v1x_hat).*normal_sf.(η_p2) .+ (2.0.*(h1x .- w .* γ).* v1x_hat.^0.5 .+ η_p2 .* v1x_hat ).*normal_pdf.(η_p2)
                            .+
                            ((h1x .+ w .* γ).^2.0 .+ v1x_hat).*normal_cdf.(η_m2) .- (2.0.*(h1x .+ w .* γ).* v1x_hat.^0.5 .+ η_m2 .* v1x_hat ).*normal_pdf.(η_m2)
                        ) ./ q1x_hat .^ 2.0
                        .-
                        x1_hat.^2.0
                    ),
                clamp_min, clamp_max
            )

            ## z
            if debug
                println("(z)")
            end
            # z1_hat .= (h1z .+ y) ./ (1.0 .+ q1z_hat)
            # chi1z .= clamp.(
            #     1.0 ./ (1.0 .+ q1z_hat),
            #     clamp_min, clamp_max
            # )
            # v1z .= clamp.(
                # ((h1z .+ y)./(1.0 .+ q1z_hat)).^2.0 
                #     .+ 
                #     v1z_hat ./　(1.0 .+ q1z_hat).^2.0 
            #     clamp_min, clamp_max
            # )
            Threads.@threads for j in 1:m
                if info && j==1
                    println("z1_hat")
                end
                z1_hat[j] = poisson_expect_sc(
                    c-> (h1z[j] + c*y[j]) / (c + q1z_hat),
                    μ, c_max
                )
                if info && j==1
                    println("chi1z")
                end
                chi1z_temp[j] = poisson_expect_sc(
                    c->1.0/(c + q1z_hat),
                    μ, c_max
                )
                if info && j==1
                    println("v1z")
                end
                v1z_temp[j] = poisson_expect_sc(
                        c->(
                            ((h1z[j] + c*y[j])./(c + q1z_hat))^2.0 
                            + 
                            v1z_hat /　(c + q1z_hat)^2.0 
                        ), μ, c_max
                ) - z1_hat[j]^2.0
            end
            chi1z = mean(chi1z_temp)
            v1z = mean(v1z_temp)

            # message passing (factorized -> Gaussian)
            if debug
                println("message passing (factorized -> Gaussian)")
            end
            h2x .= x1_hat ./ chi1x .- h1x
            q2x_hat = clamp.(
                1.0 / chi1x - q1x_hat,
                clamp_min, clamp_max
            )
            v2x_hat = clamp.(
                v1x / chi1x^2.0 - v1x_hat,
                clamp_min, clamp_max
            )
            h2z .= z1_hat ./ chi1z .- h1z
            q2z_hat = clamp.(
                1.0 / chi1z - q1z_hat,
                clamp_min, clamp_max
            )
            v2z_hat = clamp.(
                v1z / chi1z^2.0 - v1z_hat,
                clamp_min, clamp_max
            )
            
            # Gaussian part
            if debug
                println("Gaussian part")
            end
            # # ---- naive imprementation ----
            # X .= inv(I(n).*q2x_hat .+ J .* q2z_hat)
            # Y .= I(n) .* v2x_hat .+ J .* v2z_hat
            # ## x
            # if debug
            #     println("(x)")
            # end
            # x2_hat .= X * (h2x .+ A'*h2z)
            # chi2x = clamp.(
            #     mean(diag(X)),
            #     clamp_min, clamp_max
            # )
            # v2x = clamp.(
            #     mean(diag(X * Y * X)),
            #     clamp_min, clamp_max
            # )
            # ## z
            # if debug
            #     println("(z)")
            # end
            # z2_hat .= A * x2_hat
            # chi2z = clamp.(
            #     mean(diag(A*X*A')),
            #     clamp_min, clamp_max
            # )
            # v2z = clamp.(
            #     mean(diag(A*X*Y*X*A')),
            #     clamp_min, clamp_max
            # )


            # ----- efficient (?) implementation -----
            temp .= h2x .+ A' * h2z;
            x2_hat .= temp ./ q2x_hat .- v * ((s_m.^2.0 ./(q2z_hat.^(-1.0) .+ s_m.^2.0./q2x_hat)) .* (v' * temp)) ./ q2x_hat^2.0 
            chi2x = mean(1.0./(q2x_hat .+ s_n.^2.0 .* q2z_hat))
            v2x = mean((v2x_hat .+ v2z_hat .* s_n.^2.0) ./ (q2x_hat .+ s_n.^2.0 .* q2z_hat).^2.0)

            # ## z
            z2_hat .= A * x2_hat
            chi2z = (n/m) * mean(s_n.^2.0 ./ (q2x_hat .+ s_n.^2.0 .* q2z_hat))
            v2z = (n/m) * mean(s_n.^2.0 .* (v2x_hat .+ v2z_hat .* s_n.^2.0) ./ (q2x_hat .+ s_n.^2.0 .* q2z_hat).^2.0)

            # message passing (Gaussian -> factorized)
            h1x .= dumping .* (x2_hat ./ chi2x .- h2x) .+ (1.0 .- dumping) .* h1x
            q1x_hat = dumping * (1.0 / chi2x - q2x_hat) + (1.0 - dumping) * q1x_hat
            v1x_hat = dumping * (v2x / chi2x^2.0 - v2x_hat) + (1.0 - dumping) * v1x_hat

            ## z
            h1z .= dumping .* (z2_hat ./ chi2z .- h2z) .+ (1.0 .- dumping) .* h1z
            q1z_hat = dumping * (1.0 / chi2z - q2z_hat) + (1.0 - dumping) * q1z_hat
            v1z_hat = dumping * (v2z / chi2z^2.0 - v2z_hat) + (1.0 - dumping) * v1z_hat

            diff_x = mean((x1_hat - x2_hat).^2.0)
            diff_z = mean((z1_hat - z2_hat).^2.0)
            diff = maximum([diff_x, diff_z])

            if info
                println("t = $t, diff = $diff, μ=$μ, pw=$pw, w=$w, γ=$γ")
                println("\t $(mean(chi1x)),\t $(mean(v1x)),\t  $(mean(q1x_hat)),\t $(mean(v1x_hat))")
                println("\t $(mean(chi1z)),\t $(mean(v1z)),\t  $(mean(q1z_hat)),\t $(mean(v1z_hat))")
                println("\t $(minimum(chi1x)),\t $(minimum(v1x)),\t  $(minimum(q1x_hat)),\t $(minimum(v1x_hat))")
                println("\t $(minimum(chi1z)),\t $(minimum(v1z)),\t  $(minimum(q1z_hat)),\t $(minimum(v1z_hat))")
            end
            q1x_hat = clamp(q1x_hat, clamp_min, clamp_max)
            v1x_hat = clamp(v1x_hat, clamp_min, clamp_max)
            q1z_hat = clamp(q1z_hat, clamp_min, clamp_max)
            v1z_hat = clamp(v1z_hat, clamp_min, clamp_max)
            if diff < tol
                break
            end
        end

        # calculate Π
        η_p .= (γ .- h1x) ./ v1x_hat.^0.5
        η_m .= (-1.0).*(γ .+ h1x) ./ v1x_hat.^0.5
        η_p2 .= (w .* γ .- h1x) ./ v1x_hat.^0.5
        η_m2 .= (-1.0) .* (w .* γ .+ h1x) ./ v1x_hat.^0.5
        Π = (1.0 .- pw) .* (
            normal_sf.(η_p) .+ normal_cdf.(η_m)
        ) .+ pw .* (
            normal_sf.(η_p2) .+ normal_cdf.(η_m2)
        ) 

        q1x_hat_array[λ_index] = q1x_hat
        v1x_hat_array[λ_index] = v1x_hat
        h1x_array[λ_index, :] = h1x
        Π_array[λ_index, :] = Π
        x1_hat_array[λ_index, :] = x1_hat

        if info 
            println()
        end
        
    end

    result = RVAMPDiagonalRestricted(q1x_hat_array, v1x_hat_array, h1x_array, Π_array, x1_hat_array);
    return result
end


"""
replicated vector AMP (diagonal covariance)
logistic regression (not implemented yet)
"""
function rvamp(
        A::Array{Float64, 2}, y::Array{Float64,1}, λ::Array{Float64, 1}, family::Binomial{Float64}, cov::Diagonal;
        intercept=false, dumping=0.85, 
    )::RVAMPDiagonal
    # initialization
    η = dumping;
    m, n = size(A);

    q1x_hat_array = zeros((length(λ), n));
    v1x_hat_array = zeros((length(λ), n));
    h1x_array = zeros((length(λ), n));
    Π_array = zeros((length(λ), n));
    
    println("approximate stability selection for logistic regression (not implemented)")

    result = RVAMPDiagonal(q1x_hat_array, v1x_hat_array, h1x_array, Π_array);
    return result
end

# """
# vector AMP (diagonal covariance)
# linear regression
# """
# function vamp(
#         A::Array{Float64, 2}, y::Array{Float64,1}, λ::Array{Float64, 1}, family::Normal{Float64}, cov::Diagonal;
#         intercept=false, dumping=0.85, t_max = 30, n_points=15, tol=1e-8, 
#         clamp_min=1.0e-9, clamp_max=1.0e9, info=true
#     )::VAMPDiagonal
#     println("vamp for linear regression")
#     println("(diagonal covariance)")

#     # preparation
#     m, n = size(A);  # system size

#     q1x_hat_array = zeros((length(λ), n));
#     h1x_array = zeros((length(λ), n));
#     x1_hat_array = zeros((length(λ), n));

#     h1x = rand(Normal(0.0,1.0), n);
#     h2x = zeros(n);
#     q1x_hat = ones(n);
#     q2x_hat = zeros(n);
#     x1_hat = zeros(n);
#     x2_hat = zeros(n);
#     chi1x = zeros(n);
#     chi2x = zeros(n);

#     h1z = rand(Normal(0.0,1.0), m);
#     h2z = zeros(m);
#     q1z_hat = ones(m);
#     q2z_hat = zeros(m);
#     z1_hat = zeros(m);
#     z2_hat = zeros(m);
#     chi1z = zeros(m);
#     chi2z = zeros(m);

#     X = zeros((n, n));
    

#     for (λ_index, γ) in enumerate(λ)
#         if info
#             println("$λ_index, $γ")
#         end
        
#         for t in 1:t_max
#             # factorized part
#             ## x
#             x1_hat .= (h1x .- γ.* sign.(h1x)) .* heaviside.(h1x, γ)./ q1x_hat
#             chi1x .= clamp.(heaviside.(h1x, γ) ./ q1x_hat, clamp_min, clamp_max)
#             ## z
#             z1_hat .= (h1z .+ y) ./ (1.0 .+ q1z_hat)
#             chi1z .= 1.0 ./ (1.0 .+ q1z_hat)

#             # message passing (factorzied -> Gaussian)
#             ## x
#             h2x .= x1_hat ./ chi1x .- h1x
#             q2x_hat .= clamp.(1.0 ./ chi1x .- q1x_hat, clamp_min, clamp_max)
#             ## z
#             h2z .= z1_hat ./ chi1z .- h1z
#             q2z_hat .= clamp.(1.0 ./ chi1z .- q1z_hat, clamp_min, clamp_max)

#             # Gaussian part
#             X .= inv(diagm(q2x_hat) + A' * diagm(q2z_hat) * A)
#             ## x
#             x2_hat .= X * (h2x + A'* h2z)
#             chi2x .= diag(X)
#             ## z
#             z2_hat .= A * x2_hat
#             chi2z .= diag(A*X*A')

#             # message passing (Gaussian -> factorized)
#             ## x
#             h1x .= dumping .* (x2_hat ./ chi2x .- h2x) .+ (1.0 .- dumping) .* h1x
#             q1x_hat .= dumping .* (1.0 ./ chi2x .- q2x_hat) .+ (1.0 .- dumping) .* q1x_hat

#             ## z
#             h1z .= dumping .* (z2_hat ./ chi2z .- h2z) .+ (1.0 .- dumping) .* h1z
#             q1z_hat .= dumping .* (1.0 ./ chi2z .- q2z_hat) .+ (1.0 .- dumping) .* q1z_hat


#             diff_x = mean((x1_hat - x2_hat).^2.0)
#             diff_z = mean((z1_hat - z2_hat).^2.0)
#             diff = maximum([diff_x, diff_z])

#             if info
#                 println("t = $t, diff = $diff, γ=$γ")
#                 println("\t $(mean(chi1x)), $(mean(q1x_hat)), $(mean(chi2x)), $(mean(q2x_hat))")
#                 println("\t $(mean(chi1z)), $(mean(q1z_hat)), $(mean(chi2z)), $(mean(q2z_hat))")
#                 println("\t $(minimum(chi1x)), $(minimum(q1x_hat)), $(minimum(chi2x)), $(minimum(q2x_hat))")
#                 println("\t $(minimum(chi1z)), $(minimum(q1z_hat)), $(minimum(chi2z)), $(minimum(q2z_hat))")
#             end
#         end

#         q1x_hat_array[λ_index, :] = q1x_hat
#         h1x_array[λ_index, :] = h1x
#         x1_hat_array[λ_index, :] = x1_hat

#         if info 
#             println()
#         end
#     end
#     result = VAMPDiagonal(q1x_hat_array, h1x_array, x1_hat_array)

#     return result
# end

# """
# vector AMP (diagonal restricted covariance)
# linear regression
# """
# function vamp(
#         A::Array{Float64, 2}, y::Array{Float64,1}, λ::Array{Float64, 1}, family::Normal{Float64}, cov::DiagonalRestricted;
#         intercept=false, dumping=0.85, t_max = 30, n_points=15, tol=1e-8, 
#         clamp_min=1.0e-9, clamp_max=1.0e9, info=true
#     )::VAMPDiagonalRestricted
#     println("vamp for linear regression")
#     println("(diagonal covariance)")

#     # preparation
#     m, n = size(A);  # system size

#     q1x_hat_array = zeros((length(λ)));
#     h1x_array = zeros((length(λ), n));
#     x1_hat_array = zeros((length(λ), n));

#     h1x = rand(Normal(0.0,1.0), n);
#     h2x = zeros(n);
#     q1x_hat = 1.0;
#     q2x_hat = 1.0;
#     x1_hat = zeros(n);
#     x2_hat = zeros(n);
#     chi1x = 1.0;
#     chi2x = 1.0;

#     h1z = rand(Normal(0.0,1.0), m);
#     h2z = zeros(m);
#     q1z_hat = 1.0;
#     q2z_hat = 1.0;
#     z1_hat = zeros(m);
#     z2_hat = zeros(m);
#     chi1z = 1.0;
#     chi2z = 1.0;

#     X = zeros((n, n));
#     J = A'*A

#     for (λ_index, γ) in enumerate(λ)
#         if info
#             println("$λ_index, $γ")
#         end
        
#         for t in 1:t_max
#             # factorized part
#             ## x
#             x1_hat .= (h1x .- γ.* sign.(h1x)) .* heaviside.(h1x, γ)./ q1x_hat
#             chi1x = mean(heaviside.(h1x, γ) ./ q1x_hat)
#             ## z
#             z1_hat .= (h1z .+ y) ./ (1.0 .+ q1z_hat)
#             chi1z = mean(1.0 ./ (1.0 .+ q1z_hat))

#             # message passing (factorzied -> Gaussian)
#             ## x
#             h2x .= x1_hat ./ chi1x .- h1x
#             q2x_hat = clamp.(1.0 ./ chi1x .- q1x_hat, clamp_min, clamp_max)
#             ## z
#             h2z .= z1_hat ./ chi1z .- h1z
#             q2z_hat = clamp.(1.0 ./ chi1z .- q1z_hat, clamp_min, clamp_max)

#             # Gaussian part
#             X .= inv(I(n).*q2x_hat .+ J .* q2z_hat)
#             ## x
#             x2_hat .= X * (h2x .+ A'* h2z)
#             chi2x = mean(diag(X))
#             ## z
#             z2_hat .= A * x2_hat
#             chi2z = mean(diag(A*X*A'))

#             # message passing (Gaussian -> factorized)
#             ## x
#             h1x .= dumping .* (x2_hat ./ chi2x .- h2x) .+ (1.0 .- dumping) .* h1x
#             q1x_hat = dumping .* (1.0 ./ chi2x .- q2x_hat) .+ (1.0 .- dumping) .* q1x_hat

#             ## z
#             h1z .= dumping .* (z2_hat ./ chi2z .- h2z) .+ (1.0 .- dumping) .* h1z
#             q1z_hat = dumping .* (1.0 ./ chi2z .- q2z_hat) .+ (1.0 .- dumping) .* q1z_hat


#             diff_x = mean((x1_hat - x2_hat).^2.0)
#             diff_z = mean((z1_hat - z2_hat).^2.0)
#             diff = maximum([diff_x, diff_z])

#             if info
#                 println("t = $t, diff = $diff, γ=$γ")
#                 println("\t $(mean(chi1x)), $(mean(q1x_hat)), $(mean(chi2x)), $(mean(q2x_hat))")
#                 println("\t $(mean(chi1z)), $(mean(q1z_hat)), $(mean(chi2z)), $(mean(q2z_hat))")
#                 println("\t $(minimum(chi1x)), $(minimum(q1x_hat)), $(minimum(chi2x)), $(minimum(q2x_hat))")
#                 println("\t $(minimum(chi1z)), $(minimum(q1z_hat)), $(minimum(chi2z)), $(minimum(q2z_hat))")
#             end
#         end

#         q1x_hat_array[λ_index] = q1x_hat
#         h1x_array[λ_index, :] = h1x
#         x1_hat_array[λ_index, :] = x1_hat

#         if info 
#             println()
#         end
#     end
#     result = VAMPDiagonalRestricted(q1x_hat_array, h1x_array, x1_hat_array)

#     return result
# end


end  # end module 