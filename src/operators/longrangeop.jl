# long-range operators with exponentially decaying couplings, in Schur (W-form) MPO
# representation (aligned with TEMPO's schurmpo exponential-decay operators).
#
#   ExpDecayOpTerm(a, m, b, α, λ)  represents the operator
#       O = α Σ_{i<j} λ^(j-i) · a_i m_(i+1) m_(i+2) ⋯ m_(j-1) b_j ,
#   i.e. the operator `a` acts at the start site `i`, `b` at the end site `j`, the
#   propagator `m` acts on every site in between, and each propagation step contributes
#   a decay factor `λ` (total λ^(j-i), so the term strength decays exponentially with
#   the distance j - i). The whole sum carries an overall strength `α`.
#
#   ExpDecayOpSum(a, m, b, αs, λs) is the sum of |αs| such terms, one per parameter
#   pair (αs[k], λs[k]).
#
# Both admit an exact W-form (SchurMPOTensor) representation with one internal channel
# per decay parameter: the channel is opened by `C = α·λ·a`, propagated as `A = λ·m`,
# and closed by `B = b`; each channel therefore accumulates exactly the factor
# α·λ^(j-i)·m^(j-i-1) between the two endpoints.
#
#   PairOpTerm(a, m, b, C) represents the *pair-coupled* long-range operator
#       O = Σ_{i<j} C[i,j] · a_i m_(i+1) ⋯ m_(j-1) b_j + Σ_i C[i,i] · hab_i
#   with an arbitrary upper-triangular coupling matrix `C` — the off-diagonal entries
#   the pair couplings with no constraint on their distance dependence (power-law,
#   random, …), the diagonal entries the on-site products `hab` (default `a·b`). The
#   5-argument form propagates the channel with `m` between the endpoints (e.g. a
#   Jordan-Wigner string for fermionic `a_i† a_j` couplings); the 4-argument form takes
#   `m = I`. This is the finite-state-machine MPO class of TEMPO's `_fsm_mpohamiltonian`
#   (identity propagation) generalized by a uniform channel propagator: one channel per
#   coupled starting site, opened with `a`, propagated with `m`, closed with `C[i,j]·b`
#   — bond dimension L + 1. The identity-propagator ExpDecayOpSum is the special case
#   C[i,j] = Σ_k α_k λ_k^(j-i) (zero diagonal).

"""
	ExpDecayOpTerm(a, m, b, α, λ)
	ExpDecayOpTerm(a, b, α, λ)

An exponentially decaying long-range operator term
`O = α Σ_{i<j} λ^(j-i) · a_i m_(i+1) ⋯ m_(j-1) b_j` (see the module docstring). `a`,
`m`, `b` are `d×d` matrices, `λ` the decay factor per lattice spacing and `α` the
overall strength. The 4-argument form takes the propagator `m` to be the identity.
Converts directly to a [`SchurMPOTensor`](@ref).
"""
struct ExpDecayOpTerm{T<:Number}
	a::Matrix{T}
	m::Matrix{T}
	b::Matrix{T}
	α::T
	λ::T
	function ExpDecayOpTerm(a::AbstractMatrix, m::AbstractMatrix, b::AbstractMatrix,
							α::Number, λ::Number)
		d = size(a, 1)
		(size(a, 2) == size(m, 1) == size(m, 2) == size(b, 1) == size(b, 2) == d) ||
			throw(DimensionMismatch("a, m, b must be square matrices of equal size"))
		T = promote_type(typeof(α), typeof(λ), scalartype(a), scalartype(m), scalartype(b))
		return new{T}(convert(Matrix{T}, a), convert(Matrix{T}, m),
			convert(Matrix{T}, b), convert(T, α), convert(T, λ))
	end
end

scalartype(::Type{<:ExpDecayOpTerm{T}}) where {T} = T
phydim(t::ExpDecayOpTerm) = size(t.a, 1)

# identity-propagator convenience: ExpDecayOpTerm(a, b, α, λ)
ExpDecayOpTerm(a::AbstractMatrix, b::AbstractMatrix, α::Number, λ::Number) =
	ExpDecayOpTerm(a, one(a), b, α, λ)

# the hermitian conjugate: adjointed operators, conjugated couplings (λ^(j-i) conj(λ)^
# under the adjoint is carried by the adjointed propagator/endpoint chain)
Base.adjoint(t::ExpDecayOpTerm) =
	ExpDecayOpTerm(adjoint(t.a), adjoint(t.m), adjoint(t.b), conj(t.α), conj(t.λ))

"""
	ExpDecayOpSum(a, m, b, αs, λs)
	ExpDecayOpSum(a, b, αs, λs)

The sum of the exponentially decaying terms [`ExpDecayOpTerm(a, m, b, αs[k], λs[k])`](@ref
ExpDecayOpTerm) over all parameter pairs (αs[k], λs[k]). The 4-argument form takes the
propagator `m` to be the identity. Converts directly to a
[`SchurMPOTensor`](@ref) with one internal channel per pair.
"""
struct ExpDecayOpSum{T<:Number}
	a::Matrix{T}
	m::Matrix{T}
	b::Matrix{T}
	αs::Vector{T}
	λs::Vector{T}
	function ExpDecayOpSum(a::AbstractMatrix, m::AbstractMatrix, b::AbstractMatrix,
						   αs::AbstractVector, λs::AbstractVector)
		d = size(a, 1)
		(size(a, 2) == size(m, 1) == size(m, 2) == size(b, 1) == size(b, 2) == d) ||
			throw(DimensionMismatch("a, m, b must be square matrices of equal size"))
		(length(αs) == length(λs) && !isempty(αs)) ||
			throw(ArgumentError("αs and λs must be non-empty vectors of equal length"))
		T = promote_type(scalartype(a), scalartype(m), scalartype(b))
		for v in (αs..., λs...)
			T = promote_type(T, typeof(v))
		end
		return new{T}(convert(Matrix{T}, a), convert(Matrix{T}, m),
			convert(Matrix{T}, b), convert(Vector{T}, collect(αs)),
			convert(Vector{T}, collect(λs)))
	end
end

ExpDecayOpSum(t::ExpDecayOpTerm) = ExpDecayOpSum(t.a, t.m, t.b, [t.α], [t.λ])
# identity-propagator convenience: ExpDecayOpSum(a, b, αs, λs)
ExpDecayOpSum(a::AbstractMatrix, b::AbstractMatrix, αs::AbstractVector, λs::AbstractVector) =
	ExpDecayOpSum(a, one(a), b, αs, λs)

Base.adjoint(s::ExpDecayOpSum) =
	ExpDecayOpSum(adjoint(s.a), adjoint(s.m), adjoint(s.b), conj.(s.αs), conj.(s.λs))
scalartype(::Type{<:ExpDecayOpSum{T}}) where {T} = T
phydim(s::ExpDecayOpSum) = size(s.a, 1)

# the W-form site tensor of an ExpDecayOpSum: n internal channels (one per parameter
# pair), C[k] = αk·λk·a (start, carrying the first decay factor), A[k,k] = λk·m (each
# further propagation step), B[k] = b (close); a term spanning i < j therefore
# accumulates αk·λk^(j-i) · a ⊗ m^⊗(j-i-1) ⊗ b exactly. The optional on-site operator
# `hloc` accumulates in the D corner (every site). The boundary sites keep only the
# vacuum row (the starts) and the closing column (the closes) — the
# `MPOHamiltonian(::Vector{<:SchurMPOTensor})` constructor handles this through the
# row/column selection of `tompotensors`.
function SchurMPOTensor(s::ExpDecayOpSum{T}, hloc::AbstractMatrix = zeros(T, phydim(s), phydim(s))) where {T}
	d = phydim(s)
	n = length(s.αs)
	(size(hloc, 1) == size(hloc, 2) == d) ||
		throw(DimensionMismatch("hloc must be a $(d)×$(d) matrix"))
	logical = Matrix{Any}(undef, n + 2, n + 2)
	for I in eachindex(logical)
		logical[I] = 0.0
	end
	logical[1, 1] = logical[n + 2, n + 2] = 1   # the implied identity corners
	for k in 1:n
		logical[1, k+1] = s.αs[k] * s.λs[k] * s.a    # C: vacuum -> channel k
		logical[k+1, k+1] = s.λs[k] * s.m            # A: propagate channel k
		logical[k+1, n+2] = s.b                      # B: channel k -> closing
	end
	logical[1, n+2] = hloc                           # D: on-site term
	return SchurMPOTensor{T}(logical)
end
SchurMPOTensor(t::ExpDecayOpTerm, hloc::AbstractMatrix = zeros(scalartype(t), phydim(t), phydim(t))) =
	SchurMPOTensor(ExpDecayOpSum(t), hloc)

"""
	MPOHamiltonian(s::ExpDecayOpSum, L::Int, [hloc]) -> MPOHamiltonian
	MPOHamiltonian(t::ExpDecayOpTerm, L::Int, [hloc]) -> MPOHamiltonian

The exponentially decaying operator as an `L`-site Hamiltonian chain (assembled from the
per-site [`SchurMPOTensor`](@ref)s of `s`, with the optional on-site term `hloc` in
every `D` corner).
"""
MPOHamiltonian(s::ExpDecayOpSum, L::Int, hloc::AbstractMatrix = zeros(scalartype(s), phydim(s), phydim(s))) =
	MPOHamiltonian([SchurMPOTensor(s, hloc) for _ in 1:L])
MPOHamiltonian(t::ExpDecayOpTerm, L::Int, hloc::AbstractMatrix = zeros(scalartype(t), phydim(t), phydim(t))) =
	MPOHamiltonian(ExpDecayOpSum(t), L, hloc)

"""
	MPOHamiltonian(s::ExpDecayOpSum, hlocs::AbstractVector{<:AbstractMatrix}) -> MPOHamiltonian

The exponentially decaying operator on a chain whose length is `length(hlocs)`, with a
**per-site** on-site operator `hlocs[i]` in the `D` corner of site `i` (each `hlocs[i]`
must match the common local dimension `phydim(s)`; the operator content may differ from
site to site).
"""
MPOHamiltonian(s::ExpDecayOpSum, hlocs::AbstractVector{<:AbstractMatrix}) =
	MPOHamiltonian([SchurMPOTensor(s, hloc) for hloc in hlocs])

# debug constructor: expand the exact W-form representation into the (typically much
# larger) explicit OpSum of all two-site terms
function OpSum(s::ExpDecayOpSum, L::Int)
	d = phydim(s)
	terms = OpTerm[]
	for k in eachindex(s.αs), i in 1:L-1, j in i+1:L
		positions = collect(i:j)
		operators = [ifelse(t == i, s.a, ifelse(t == j, s.b, s.m)) for t in positions]
		push!(terms, OpTerm(s.αs[k] * s.λs[k]^(j - i), positions, operators))
	end
	return OpSum(fill(d, L), terms)
end
OpSum(t::ExpDecayOpTerm, L::Int) = OpSum(ExpDecayOpSum(t), L)

"""
	PairOpTerm(a, m, b, C, [hab])
	PairOpTerm(a, b, C, [hab])

A pair-coupled long-range operator term with an **arbitrary** coupling matrix (see the
module docstring): `O = Σ_{i<j} C[i,j] · a_i m_(i+1) ⋯ m_(j-1) b_j + Σ_i C[i,i] · hab_i`.
`a`, `m`, `b` are `d×d` matrices and `C` an upper triangular `L×L` coupling matrix —
the off-diagonal `C[i,j]` (`i < j`) the pair couplings, the diagonal `C[i,i]` the
on-site operators `hab` (default `a·b`; stored explicitly because its adjoint is
`b†·a†`, not expressible through adjointed endpoints when `a` and `b` do not commute).
The 4-argument form takes the propagator `m` to be the identity; with a Jordan-Wigner
string as `m` the term represents fermionic `a_i† a_j`-type couplings exactly. Converts
to per-site [`SchurMPOTensor`](@ref)s in the finite-state-machine layout (one channel
per coupled starting site, bond dimension `L + 1`).
"""
struct PairOpTerm{T<:Number}
	a::Matrix{T}
	m::Matrix{T}
	b::Matrix{T}
	hab::Matrix{T}
	C::Matrix{T}
	function PairOpTerm(a::AbstractMatrix, m::AbstractMatrix, b::AbstractMatrix,
						C::AbstractMatrix, hab::AbstractMatrix = a * b)
		d = size(a, 1)
		(size(a, 2) == size(m, 1) == size(m, 2) == size(b, 1) == size(b, 2) == d) ||
			throw(DimensionMismatch("a, m, b must be square matrices of equal size"))
		(size(hab, 1) == size(hab, 2) == d) ||
			throw(DimensionMismatch("hab must be a $(d)×$(d) matrix"))
		L = size(C, 1)
		(L == size(C, 2) && L >= 2) ||
			throw(ArgumentError("C must be a square L×L coupling matrix with L ≥ 2"))
		istriu(C) || throw(ArgumentError("C must be upper triangular"))
		T = promote_type(scalartype(a), scalartype(m), scalartype(b), scalartype(hab),
						 scalartype(C))
		return new{T}(convert(Matrix{T}, a), convert(Matrix{T}, m), convert(Matrix{T}, b),
			convert(Matrix{T}, hab), convert(Matrix{T}, C))
	end
end

# identity-propagator convenience: PairOpTerm(a, b, C, [hab])
PairOpTerm(a::AbstractMatrix, b::AbstractMatrix, C::AbstractMatrix,
		   hab::AbstractMatrix = a * b) = PairOpTerm(a, one(a), b, C, hab)

Base.length(t::PairOpTerm) = size(t.C, 1)
scalartype(::Type{<:PairOpTerm{T}}) where {T} = T
phydim(t::PairOpTerm) = size(t.a, 1)

# the hermitian conjugate: adjointed operators (in place), conjugated couplings and the
# adjointed on-site product
Base.adjoint(t::PairOpTerm) =
	PairOpTerm(adjoint(t.a), adjoint(t.m), adjoint(t.b), conj(t.C), adjoint(t.hab))

# the W-form site tensor of a PairOpTerm at site `s` (site-dependent): the
# finite-state-machine layout of TEMPO's `_fsm_mpohamiltonian` generalized by a uniform
# channel propagator `m`. Channel k = the starting site k: opened with `a` (the C block,
# only if channel k couples to some later site), propagated with `m` while
# k < s < lastclose[k] (the A blocks), closed with `C[k,s]·b` (the B block). The coupling
# sits entirely at the closing site, so `C` is completely arbitrary; the D corner merges
# the on-site diagonal coupling `C[s,s]·hab` with the external on-site operator `hloc`
# (every site). Boundary sites keep only the vacuum row and the closing column through
# the row/column selection of `tompotensors`.
function SchurMPOTensor(t::PairOpTerm{T}, s::Int,
						hloc::AbstractMatrix = zeros(T, phydim(t), phydim(t))) where {T}
	d = phydim(t)
	L = length(t)
	(1 <= s <= L) || throw(ArgumentError("site out of range: $s not in 1:$L"))
	(size(hloc, 1) == size(hloc, 2) == d) ||
		throw(DimensionMismatch("hloc must be a $(d)×$(d) matrix"))
	# channel k stays alive up to its last coupled site (0 if never: dead channel)
	lastclose = zeros(Int, L - 1)
	for k in 1:L-1, j in k+1:L
		!iszero(t.C[k, j]) && (lastclose[k] = j)
	end
	logical = Matrix{Any}(undef, L + 1, L + 1)
	for I in eachindex(logical)
		logical[I] = 0.0
	end
	logical[1, 1] = logical[L + 1, L + 1] = 1   # the implied identity corners
	for k in 1:L-1
		k == s && lastclose[k] > k && (logical[1, k+1] = t.a)    # C: open channel k
		k < s < lastclose[k] && (logical[k+1, k+1] = t.m)       # A: propagate alive channel
		k < s && !iszero(t.C[k, s]) && (logical[k+1, L+1] = t.C[k, s] * t.b)  # B: close
	end
	logical[1, L+1] = t.C[s, s] * t.hab + hloc   # D: diagonal coupling + on-site term
	# the explicit physical dimension keeps all-scalar tensors (vanishing C and hloc) valid
	return SchurMPOTensor{T}(logical, d)
end

"""
	MPOHamiltonian(t::PairOpTerm, [hloc]) -> MPOHamiltonian
	MPOHamiltonian(t::PairOpTerm, hlocs::AbstractVector{<:AbstractMatrix}) -> MPOHamiltonian

The pair-coupled operator as an `L`-site Hamiltonian chain (`L = length(t)`), assembled
from the per-site [`SchurMPOTensor`](@ref)s. The optional on-site operator `hloc` (the
same on every site) — or the per-site operators `hlocs` (each `hlocs[s]` at site `s`,
operator content may differ) — merges into the `D` corners with the diagonal couplings
`C[s,s]·hab`.
"""
MPOHamiltonian(t::PairOpTerm{T}, hloc::AbstractMatrix = zeros(T, phydim(t), phydim(t))) where {T} =
	MPOHamiltonian([SchurMPOTensor(t, s, hloc) for s in 1:length(t)])
function MPOHamiltonian(t::PairOpTerm, hlocs::AbstractVector{<:AbstractMatrix})
	length(hlocs) == length(t) ||
		throw(DimensionMismatch("need one hloc per site ($(length(t))), got $(length(hlocs))"))
	return MPOHamiltonian([SchurMPOTensor(t, s, hlocs[s]) for s in 1:length(t)])
end

# debug constructor: expand the exact finite-state-machine representation into the
# (typically much larger) explicit OpSum of all terms
function OpSum(t::PairOpTerm)
	L, d = length(t), phydim(t)
	terms = OpTerm[]
	for i in 1:L-1, j in i+1:L
		iszero(t.C[i, j]) && continue
		positions = collect(i:j)
		operators = [ifelse(tt == i, t.a, ifelse(tt == j, t.b, t.m)) for tt in positions]
		push!(terms, OpTerm(t.C[i, j], positions, operators))
	end
	for i in 1:L
		iszero(t.C[i, i]) && continue
		push!(terms, OpTerm(t.C[i, i], i => t.hab))
	end
	return OpSum(fill(d, L), terms)
end
