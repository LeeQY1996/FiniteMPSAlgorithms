using Test
using LinearAlgebra
using Logging
using Random
using TensorOperations
using FiniteMPSAlgorithms
# not exported: access the settings module via `using FiniteMPSAlgorithms.Defaults`
using FiniteMPSAlgorithms.Defaults
# disambiguate from LinearAlgebra's SVD/QR/LQ factorization objects
using FiniteMPSAlgorithms: SVD, QR, QRpos, LQ, LQpos, SDD, Polar, max_bonddims, l_LL, r_RR,
                           PauliTerm

include("helpers.jl")

@testset verbose = true "FiniteMPSAlgorithms" begin
	include("shared_core.jl")
	# The core tensor/state/MPO suites live with MOSSQuantumMPS. These
	# cases exercise algorithms against MPOs and the model builders.
	@testset "operator and model integration" begin
		include("operators/longrangeop.jl")
		include("operators/sparsempo.jl")
		include("operators/vectorize.jl")
		include("operators/models.jl")
		include("operators/nonuniform.jl")
	end
	@testset "algorithms" begin
		include("algorithms/arithmetics.jl")
		include("algorithms/dmrg.jl")
		include("algorithms/timeevo.jl")
		include("algorithms/tebd.jl")
		include("algorithms/seq2seq.jl")
		include("algorithms/sampling.jl")
		include("algorithms/experimental.jl")
	end
end
