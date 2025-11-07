# EmergentMasterControl.jl (FINAL, VERIFIED)
module EmergentMasterControl

using LinearAlgebra, Statistics, Random, JSON3, Dates

struct SpecialistEntity
    id::String
    dimensionality::Int
    w_feature::Matrix{Float64}
    b_feature::Vector{Float64}
    w_score::Matrix{Float64}
    b_score::Vector{Float64}
    w_proj_to_latent::Matrix{Float64}
    b_proj_to_latent::Vector{Float64}
    w_proj_from_latent::Matrix{Float64}
    b_proj_from_latent::Vector{Float64}
end

struct MasterControl
    id::String
    pantheon::Dict{Int, SpecialistEntity}
    latent_dim::Int

    # This is the "Main" inner constructor that loads from a file
    function MasterControl(weights_file::String = "EAMC_weights.json")
        println("🏛️  Instantiating the Emergent Architecture Master Control...")
        data = JSON3.read(read(weights_file, String))
        pantheon = Dict{Int, SpecialistEntity}()
        latent_dim = data.latent_dim
        for (key, specialist_data) in data.pantheon
            dim = specialist_data.dimensionality
            w_f = Matrix(hcat(specialist_data.weights.feature_extractor.W...)')
            b_f = Vector{Float64}(specialist_data.weights.feature_extractor.b)
            w_s = Matrix(hcat(specialist_data.weights.scoring_head.W...)')
            b_s = Vector{Float64}(specialist_data.weights.scoring_head.b)
            w_ptl = Matrix(hcat(specialist_data.weights.project_to_latent.W...)')
            b_ptl = Vector{Float64}(specialist_data.weights.project_to_latent.b)
            w_pfl = Matrix(hcat(specialist_data.weights.project_from_latent.W...)')
            b_pfl = Vector{Float64}(specialist_data.weights.project_from_latent.b)
            id = "specialist-$(dim)D-$(randstring(4))"
            pantheon[dim] = SpecialistEntity(id, dim, w_f, b_f, w_s, b_s, w_ptl, b_ptl, w_pfl, b_pfl)
            println("   - ✅ Loaded $(dim)D Specialist into the Pantheon.")
        end
        new("EAMC-v1-$(randstring(6))", pantheon, latent_dim)
    end

    # --- THIS IS THE FIX ---
    # This is a second, "internal" constructor for creating sub-councils during testing.
    # It takes the data directly instead of reading from a file.
    function MasterControl(id::String, pantheon::Dict{Int, SpecialistEntity}, latent_dim::Int)
        new(id, pantheon, latent_dim)
    end
end

# --- The Collaborative Reasoning Function (Unchanged and Correct) ---
function solve_with_pantheon(eamc::MasterControl, problem_points::Matrix{Float64})
    primary_dim = size(problem_points, 2)
    @assert haskey(eamc.pantheon, primary_dim) "No specialist in the Pantheon for $(primary_dim)D problems!"
    primary_specialist = eamc.pantheon[primary_dim]

    latent_projections = []
    initial_thoughts_map = Dict{Int, Matrix{Float64}}()
    for (dim, specialist) in eamc.pantheon
        specialist_view_of_problem = prepare_data_for_specialist(specialist, problem_points)
        features = specialist_view_of_problem * specialist.w_feature' .+ specialist.b_feature'
        activated_features = @. features / (1 + exp(-features))
        if dim == primary_dim
            initial_thoughts_map[dim] = activated_features
        end
        latent_vec = activated_features * specialist.w_proj_to_latent' .+ specialist.b_proj_to_latent'
        push!(latent_projections, latent_vec)
    end
    
    Q = K = V = vcat(latent_projections...)
    attention_scores = (Q * K') ./ sqrt(eamc.latent_dim)
    attention_weights = exp.(attention_scores .- maximum(attention_scores, dims=2))
    attention_weights ./= sum(attention_weights, dims=2)
    fused_insights = attention_weights * V
    primary_insight_slice = fused_insights[1:size(problem_points, 1), :]
    
    integrated_features = primary_insight_slice * primary_specialist.w_proj_from_latent' .+ primary_specialist.b_proj_from_latent'
    final_reasoning_features = initial_thoughts_map[primary_dim] + integrated_features
    final_scores = final_reasoning_features * primary_specialist.w_score' .+ primary_specialist.b_score'
    final_probabilities = vec(exp.(final_scores .- maximum(final_scores)) ./ sum(exp.(final_scores .- maximum(final_scores))))
    return (solution=argmax(final_probabilities), confidence=maximum(final_probabilities))
end

function prepare_data_for_specialist(specialist::SpecialistEntity, original_points::Matrix{Float64})
    original_dim = size(original_points, 2)
    target_dim = specialist.dimensionality
    if original_dim == target_dim
        return original_points
    elseif original_dim > target_dim
        return original_points[:, 1:target_dim]
    else
        return hcat(original_points, zeros(Float64, size(original_points, 1), target_dim - original_dim))
    end
end

end # module