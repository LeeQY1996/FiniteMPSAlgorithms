# TaylorCluster: N-th order time-evolution MPO from the Taylor cluster expansion.
# Ported from MPSKit (algorithms/timestep/taylorcluster.jl) to the Schur-form blocks
# of this package; reference: Van Damme et al., SciPost Phys. 17, 135 (2024).

"""
	TaylorCluster(; N=2, extension=true, compression=false)

Algorithm constructing the `N`-th order time-evolution MPO using the Taylor cluster
expansion ([`timeevompo`](@ref) builds the MPO, which is then applied like the
[`WI`](@ref)/[`WII`](@ref) steppers). The virtual channels of the evolved MPO are the
Tucker-symmetrized `N`-fold products of the Hamiltonian channels, fused back into the
Schur form by the cluster loop-back.

## Fields

- `N::Int`: order of the Taylor expansion;
- `extension::Bool`: incorporate the higher-order corrections from `H^(N+1)`
  (Algorithm 3 of the reference);
- `compression::Bool`: approximate compression of the corrections, accurate up to
  order `N` (Algorithm 4 of the reference).

## References

- Van Damme et al., SciPost Phys. 17, 135 (2024).
"""
@kwdef struct TaylorCluster <: GeneralStepper
	N::Int = 2
	extension::Bool = true
	compression::Bool = false
end

# ---------- helpers: N-fold auxiliary channel index bookkeeping ----------
#
# The evolved MPO site tensor lives on `N`-fold products of the Schur channels: the
# auxiliary index of each bond is an `N`-tuple of Schur channel numbers `1..V` (with
# `1` / `V` the two unit levels), linearized in column-major order — matching
# MPSKit's `LinearIndices(ntuple(Returns(sz), Val(N)))` bookkeeping.

_tuplin(α::NTuple{N,Int}, V::Int) where {N} =
	1 + sum((α[j] - 1) * V^(j - 1) for j in 1:N)
_lintup(lin::Int, V::Int, ::Val{N}) where {N} =
	ntuple(j -> ((lin - 1) ÷ V^(j - 1)) % V + 1, Val(N))
_insertafter(t::NTuple{M,Int}, pos::Int, x::Int) where {M} =
	ntuple(j -> j <= pos ? t[j] : j == pos + 1 ? x : t[j - 1], Val(M + 1))

# `H^N` for one site: `fuse_mul_mpo` in MPSKit — the logical Schur blocks multiply
# physically (same-site operator product) while the auxiliary channel indices of the
# two factors are kept (fused by concatenation, left keys then right keys).
function _site_pow(m::SchurMPOTensor, ::Val{N}, ::Type{T}) where {N,T}
	Vl, Vr = space_l(m), space_r(m)
	X = Dict{Tuple{Int,Int}, Matrix{T}}()
	for α in 1:Vl, β in 1:Vr
		blk = m[α, β]
		iszero(blk) || (X[(_tuplin((α,), Vl), _tuplin((β,), Vr))] = blk)
	end
	for k in 2:N
		Xn = Dict{Tuple{Int,Int}, Matrix{T}}()
		for (key, blk) in X
			al, br = key
			α = _lintup(al, Vl, Val(k - 1))
			β = _lintup(br, Vr, Val(k - 1))
			for γ in 1:Vl, δ in 1:Vr
				G = m[γ, δ]
				iszero(G) && continue
				nk = (_tuplin((γ, α...), Vl), _tuplin((δ, β...), Vr))
				prod = blk * G
				Xn[nk] = haskey(Xn, nk) ? Xn[nk] + prod : prod
			end
		end
		X = Xn
	end
	return X
end

# `fuse_mul_mpo(H_n, H)`: one more factor, the new factor's channels fuse in front
# (MPSKit fuses `left_virtualspace(O2) ⊗ left_virtualspace(O1)`, O2 being the new one)
function _site_fuse(X::Dict{Tuple{Int,Int},Matrix{T}}, m::SchurMPOTensor, ::Val{N},
					::Type{T}) where {N,T}
	Vl, Vr = space_l(m), space_r(m)
	Xn = Dict{Tuple{Int,Int}, Matrix{T}}()
	for (key, blk) in X
		al, br = key
		α = _lintup(al, Vl, Val(N))
		β = _lintup(br, Vr, Val(N))
		for γ in 1:Vl, δ in 1:Vr
			G = m[γ, δ]
			iszero(G) && continue
			nk = (_tuplin((γ, α...), Vl), _tuplin((δ, β...), Vr))
			prod = blk * G
			Xn[nk] = haskey(Xn, nk) ? Xn[nk] + prod : prod
		end
	end
	return Xn
end

# Algorithm 3 of Van Damme et al. (2024): incorporate the higher-order corrections
# from `H_next = H_n * H` into `H_n`. The correction is added over the *full* auxiliary
# space — it also creates blocks that H_n did not have.
function _taylor_extension!(slices, hs, Vl0, Vr0, ::Val{N}, τ::Number, ::Type{T}) where {N,T}
	for i in eachindex(slices)
		X = slices[i]
		Xn = _site_fuse(X, hs[i], Val(N), T)
		VlN, VrN = Vl0[i]^N, Vr0[i]^N
		for al in 1:VlN, br in 1:VrN
			a = _lintup(al, Vl0[i], Val(N))
			b = _lintup(br, Vr0[i], Val(N))
			all(>(1), b) || continue
			(all(in((1, Vl0[i])), a) && any(==(Vl0[i]), a)) && continue

			n1 = count(==(1), a) + 1
			n3 = count(==(Vr0[i]), b) + 1
			factor = τ * factorial(N) / (factorial(N + 1) * n1 * n3)
			iszero(factor) && continue

			for c in 1:(N + 1), d in 1:(N + 1)
				aₑ = _insertafter(a, c - 1, 1)
				bₑ = _insertafter(b, d - 1, Vr0[i])
				v = get(Xn, (_tuplin(aₑ, Vl0[i]), _tuplin(bₑ, Vr0[i])), nothing)
				v === nothing || (X[(al, br)] = get(X, (al, br), zeros(T, size(v))) +
												  factor * v)
			end
		end
	end
	return slices
end

# Algorithm 1: project the auxiliary virtual-bond directions onto the physical
# block, completing the Nth-order time-evolution MPO.
function _taylor_loopback!(slices, Vl0, Vr0, ::Val{N}, τ::Number) where {N}
	for i in eachindex(slices)
		X = slices[i]
		Vr = Vr0[i]
		VlN, VrN = Vl0[i]^N, Vr^N
		one_lin = 1
		for b_lin in 2:VrN                        # cinds_right[2:end], column-major order
			b = _lintup(b_lin, Vr, Val(N))
			b == ntuple(Returns(1), Val(N)) && continue
			all(x -> x == 1 || x == Vr, b) || continue

			a = count(==(Vr), b)
			factor = τ^a * factorial(N - a) / factorial(N)
			for al in 1:VlN                       # all left auxiliaries
				haskey(X, (al, b_lin)) || continue
				blk = X[(al, b_lin)]
				X[(al, one_lin)] = get(X, (al, one_lin), zero(blk)) + factor * blk
			end
			for (al, br) in collect(keys(X))
				# delete the rows/columns whose channel tuple equals the folded one;
				# the left and right auxiliary spaces may be rectangular (different
				# channel counts → different linearizations), so the right tuple b maps
				# to its row-space image m_b (C₁ stays 1, C₂ becomes the last row level)
				m_b = ntuple(j -> b[j] == 1 ? 1 : Vl0[i], Val(N))
				(_lintup(al, Vl0[i], Val(N)) == m_b || br == b_lin) && delete!(X, (al, br))
			end
		end
	end
	return slices
end

# Algorithm 2: collapse rows and columns that are equivalent under the
# permutation symmetry of the Taylor expansion.
function _partition_first(t::NTuple{M,Int}, sentinel::Int) where {M}
	n_match = count(==(sentinel), t)
	return ntuple(Val(M)) do j
		if j <= n_match
			_kth_match(t, sentinel, j, true)
		else
			_kth_match(t, sentinel, j - n_match, false)
		end
	end
end

function _kth_match(t::NTuple{M,Int}, sentinel::Int, k::Int, want_match::Bool) where {M}
	cnt = 0
	for x in t
		if (x == sentinel) == want_match
			cnt += 1
			cnt == k && return x
		end
	end
	return 0
end

function _taylor_remove_equivalents!(slices, Vl0, Vr0, ::Val{N}) where {N}
	for i in eachindex(slices)
		X = slices[i]
		Vl, Vr = Vl0[i], Vr0[i]
		Ll, Lr = Vl^N, Vr^N
		for c_lin in 1:Ll
			c = _lintup(c_lin, Vl, Val(N))
			s_c = _partition_first(c, 1)
			n1 = count(==(1), c)
			n3 = count(==(Vl), c)

			if n3 <= n1 && s_c != c
				s_lin = _tuplin(s_c, Vl)
				for k in 1:Lr
					if haskey(X, (c_lin, k))
						X[(s_lin, k)] = get(X, (s_lin, k), zero(X[(c_lin, k)])) +
										X[(c_lin, k)]
						delete!(X, (c_lin, k))
					end
				end
			end
		end
		for c_lin in 1:Lr
			c = _lintup(c_lin, Vr, Val(N))
			s_r = _partition_first(c, Vr)
			n1 = count(==(1), c)
			n3 = count(==(Vr), c)

			if n3 > n1 && s_r != c
				s_lin = _tuplin(s_r, Vr)
				for k in 1:Ll
					if haskey(X, (k, c_lin))
						X[(k, s_lin)] = get(X, (k, s_lin), zero(X[(k, c_lin)])) +
										X[(k, c_lin)]
						delete!(X, (k, c_lin))
					end
				end
			end
		end
	end
	return slices
end

# Algorithm 4: approximate compression — fold the right-going `V_right`
# columns back into the `1`-column with the matching Taylor weight, then drop
# the residual `V_left` rows.
function _taylor_compression!(slices, Vl0, Vr0, ::Val{N}, τ::Number) where {N}
	for i in eachindex(slices)
		X = slices[i]
		Vl, Vr = Vl0[i], Vr0[i]
		Ll, Lr = Vl^N, Vr^N
		for a_lin in 1:Lr
			a = _lintup(a_lin, Vr, Val(N))
			all(>(1), a) || continue
			n1 = count(==(Vr), a)
			n1 == 0 && continue
			b = _tuplin(map(x -> x == Vr ? 1 : x, a), Vr)
			factor = τ^n1 * factorial(N - n1) / factorial(N)
			for k in 1:Ll
				if haskey(X, (k, a_lin))
					X[(k, b)] = get(X, (k, b), zero(X[(k, a_lin)])) + factor * X[(k, a_lin)]
					delete!(X, (k, a_lin))
				end
			end
		end
		for a_lin in 1:Ll
			a = _lintup(a_lin, Vl, Val(N))
			all(>(1), a) || continue
			n1 = count(==(Vl), a)
			n1 == 0 && continue
			for k in 1:Lr
				delete!(X, (a_lin, k))
			end
		end
	end
	return slices
end

"""
	timeevompo(m::SchurMPOTensor, dt, alg::TaylorCluster)
	timeevompo(h::MPOHamiltonian, dt, alg::TaylorCluster)

Evolve a Schur-form Hamiltonian by one step `dt` with the [`TaylorCluster`](@ref)
algorithm: the `N`-th order time-evolution MPO of the Taylor cluster expansion is built
on the `N`-fold fused channel space and fused back into the Schur form. The
`SchurMPOTensor` method returns the site-tensor block dictionary; the Hamiltonian
method assembles the dense finite-chain [`MPO`](@ref) (vacuum row of the first site,
closing column of the last).
"""
function timeevompo(m::SchurMPOTensor, dt::Number, alg::TaylorCluster)
	T = promote_type(scalartype(m), typeof(dt))
	Vl, Vr = space_l(m), space_r(m)
	slice = _site_pow(m, Val(alg.N), T)
	alg.extension && _taylor_extension!([slice], [m], [Vl], [Vr], Val(alg.N), dt, T)
	_taylor_loopback!([slice], [Vl], [Vr], Val(alg.N), dt)
	_taylor_remove_equivalents!([slice], [Vl], [Vr], Val(alg.N))
	alg.compression && _taylor_compression!([slice], [Vl], [Vr], Val(alg.N), dt)
	return slice
end

function timeevompo(h::MPOHamiltonian, dt::Number, alg::TaylorCluster)
	L = length(h)
	T = promote_type(scalartype(h), typeof(dt))
	Vl0 = [space_l(h[i]) for i in 1:L]
	Vr0 = [space_r(h[i]) for i in 1:L]
	# the channel blocks that cannot connect through the chain boundaries are dead in the
	# cluster algebra (MPSKit's boundary convention drops them: the first site has a
	# single row, the last a single column) — zero them so no dead paths fuse in
	ms = [i == 1 ? SchurMPOTensor(zero.(h[i].A), zero.(h[i].B), h[i].C, h[i].D) :
		  i == L ? SchurMPOTensor(h[i].A, h[i].B, zero.(h[i].C), h[i].D) : h[i]
		 for i in 1:L]
	slices = [_site_pow(ms[i], Val(alg.N), T) for i in 1:L]
	alg.extension && _taylor_extension!(slices, ms, Vl0, Vr0, Val(alg.N), dt, T)
	_taylor_loopback!(slices, Vl0, Vr0, Val(alg.N), dt)
	_taylor_remove_equivalents!(slices, Vl0, Vr0, Val(alg.N))
	alg.compression && _taylor_compression!(slices, Vl0, Vr0, Val(alg.N), dt)

	# remove orphans: the channels kept at every bond are the nonvanishing ones present
	# on both sides (the left end keeps the vacuum channel, the right end the closing one)
	I = Vector{Vector{Int}}(undef, L + 1)
	_nonzero_rows = [Set(al for (al, br) in keys(slices[i]) if !iszero(slices[i][(al, br)]))
					 for i in 1:L]
	_nonzero_cols = [Set(br for (al, br) in keys(slices[i]) if !iszero(slices[i][(al, br)]))
					 for i in 1:L]
	I[1] = [1]
	for s in 2:L
		I[s] = sort!(collect(intersect(_nonzero_cols[s-1], _nonzero_rows[s])))
	end
	I[L+1] = [1]

	O = Vector{Array{T,4}}(undef, L)
	for i in 1:L
		d = phydim(h[i])
		O[i] = zeros(T, length(I[i]), d, length(I[i+1]), d)
		for (key, blk) in slices[i]
			iszero(blk) && continue
			r = findfirst(==(key[1]), I[i])
			c = findfirst(==(key[2]), I[i+1])
			(r === nothing || c === nothing) && continue
			O[i][r, :, c, :] .= blk
		end
	end
	return MPO(O)
end

# two-argument convenience parity with the other steppers is inherited from
# `timeevompo(h, dt) = timeevompo(h, dt, WII())`; `TaylorCluster` is always explicit.

"""
	timeevo(ψ::CanonicalMPS, h::MPOHamiltonian, dt, alg::GeneralStepper;
		trunc=Defaults.alg_trunc()) -> ψ'

[`GeneralStepper`](@ref) application path (the [`TaylorCluster`](@ref) stepper): the
evolved MPO is applied with [`mult`](@ref) under the truncation `trunc`, exactly like the
first-order steppers; `timeevo!` handles it through its `MPSAlgorithm` fallback.
"""
timeevo(ψ::CanonicalMPS, h::MPOHamiltonian, dt::Number, alg::GeneralStepper;
	trunc::TruncationScheme = Defaults.alg_trunc()) =
	mult(timeevompo(h, dt, alg), ψ, SVDCompression(trunc = trunc))
