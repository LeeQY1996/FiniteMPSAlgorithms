# compress: the unique exported interface for (truncating) compression of a single chain
#   compress(ψ::CanonicalMPS; alg) -> (CanonicalMPS, err)
#   compress(h::AbstractMPO; alg)  -> (AbstractMPO, err)
# dispatched on alg: SVDCompression (canonicalize with truncation) or DMRG1 (ALS maximizing
# the overlap between the compressed chain and the input chain)

# ALS compression cache: maximize |⟨out|inp⟩|; reuses OverlapCache
const CompressCache = OverlapCache

# ---------- zip-fused sweep kernels (pre-fused partial site tensors) ----------
#
# Each site fuses the overlap contraction `(cL · ket)` (right mirror: `(ket · cR)`) **once**
# into a fused tensor; the same fused tensor is then used twice — the local ALS target (one
# binary contraction with the opposite overlap environment) and the next overlap
# environment (one binary contraction with the conj of the updated bra). This halves the
# per-site contraction work of the sweeps (the general template contracts the ket tensor
# into the environments twice), the zip mechanism of the ALS sweeps in
# InfiniteMPSAlgorithms.

function _fuse_left(c::OverlapCache, s::Integer)
	if c.ket[s] isa MPOTensor
		return @tensor K[b, o, m, i] := c.cstorage[s][b, kL] * c.ket[s][kL, o, m, i]
	end
	return @tensor K[b, p, km] := c.cstorage[s][b, kL] * c.ket[s][kL, p, km]
end
function _fuse_right(c::OverlapCache, s::Integer)
	if c.ket[s] isa MPOTensor
		return @tensor S[kL, o, bR, i] := c.cstorage[s+1][bR, kR] * c.ket[s][kL, o, kR, i]
	end
	return @tensor S[kL, p, bR] := c.cstorage[s+1][bR, kR] * c.ket[s][kL, p, kR]
end

function _map_fused(K::AbstractArray{T,3}, c::OverlapCache, s::Integer) where {T}
	return @tensor t[b, p, bR] := K[b, p, km] * c.cstorage[s+1][bR, km]
end
function _map_fused(K::AbstractArray{T,4}, c::OverlapCache, s::Integer) where {T}
	return @tensor t[b, o, bR, i] := K[b, o, m, i] * c.cstorage[s+1][bR, m]
end
function _map_fusedR(S::AbstractArray{T,3}, c::OverlapCache, s::Integer) where {T}
	return @tensor t[bL, p, bR] := c.cstorage[s][bL, kL] * S[kL, p, bR]
end
function _map_fusedR(S::AbstractArray{T,4}, c::OverlapCache, s::Integer) where {T}
	return @tensor t[bL, o, bR, i] := c.cstorage[s][bL, kL] * S[kL, o, bR, i]
end

function _push_left_fused!(c::OverlapCache, K, s::Integer)
	bra = c.bra[s]
	if ndims(K) == 3
		c.cstorage[s+1] = @tensor h[bR, km] := conj(bra[bL, p, bR]) * K[bL, p, km]
	else
		c.cstorage[s+1] = @tensor h[bR, m] := conj(bra[bL, o, bR, i]) * K[bL, o, m, i]
	end
	return c
end
function _push_right_fused!(c::OverlapCache, S, s::Integer)
	bra = c.bra[s]
	if ndims(S) == 3
		c.cstorage[s] = @tensor h[bL, kL] := conj(bra[bL, p, bR]) * S[kL, p, bR]
	else
		c.cstorage[s] = @tensor h[bL, kL] := conj(bra[bL, o, bR, i]) * S[kL, o, bR, i]
	end
	return c
end

function leftsweep!(c::OverlapCache, alg::DMRG1)
	L = length(c.bra)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		K = _fuse_left(c, s)
		t = _map_fused(K, c, s)
		kvals[s] = norm(t)
		(alg.verbosity > 2) && _logupdate(stdout, "l2r", s, kvals[s])
		q, r = _gauge_left(t, alg.alg_orth)
		c.bra[s] = q
		c.bra[s+1] = _contract_first(c.bra[s+1], r)
		_push_left_fused!(c, K, s)
	end
	t = _map_fused(_fuse_left(c, L), c, L)
	kvals[L] = norm(t)
	(alg.verbosity > 2) && _logupdate(stdout, "l2r", L, kvals[L])
	c.bra[L] = t
	return kvals
end

function rightsweep!(c::OverlapCache, alg::DMRG1)
	L = length(c.bra)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		S = _fuse_right(c, s)
		t = _map_fusedR(S, c, s)
		kvals[k] = norm(t)
		(alg.verbosity > 2) && _logupdate(stdout, "r2l", s, kvals[k])
		k += 1
		l, q = _gauge_right(t, alg.alg_orth)
		c.bra[s] = q
		c.bra[s-1] = _contract_last(c.bra[s-1], l)
		_push_right_fused!(c, S, s)
	end
	t = _map_fusedR(_fuse_right(c, 1), c, 1)
	kvals[L] = norm(t)
	(alg.verbosity > 2) && _logupdate(stdout, "r2l", 1, kvals[L])
	c.bra[1] = t
	return kvals
end

sweep!(c::OverlapCache, alg::DMRG1) = vcat(leftsweep!(c, alg), rightsweep!(c, alg))

# ---------- initial guess & cache construction ----------

"""
	svdguess_compress(x, D::Int)

A truncating SVD re-compression of `x` (bond cap `D`) as an initial guess for the
iterative `compress` (accurate but costly compared to zero padding).
"""
function svdguess_compress(x, D::Int)
	out, _ = _canonicalize!(copy(x); alg=Orthogonalize(SVD(), truncdim(D), false))
	return x isa CanonicalMPS ? out : CanonicalMPO(out.data)
end

"""
	compress!(out, x, alg::DMRG1) -> (out, info)

Single-site variational (ALS) compression of `x` refined in place on the initial guess
`out`. The sweeps determine direction and magnitude together at the data level; the
input's external scale is attached through the per-site `scaling` field of the output
(a scaling^L power is never materialized). Returns `(out, info)` with `info`, the
[`ALSConvergenceInfo`](@ref) of the sweeps.
"""
function compress!(out, x, alg::DMRG1)
	bonddim(out) != alg.D && changebond!(out; D=alg.D)
	cache = OverlapCache(out, x)
	info = iterative_compute!(cache, alg)
	x isa Union{CanonicalMPS, CanonicalMPO} && setscaling!(out, scaling(x))
	return out, info
end

"""
	compress(ψ::CanonicalMPS, alg=SVDCompression(trunc=Defaults.alg_trunc())) -> CanonicalMPS
	compress(h::AbstractMPO, alg=...) -> CanonicalMPO
	compress(x, alg::DMRG1) -> (chain, info)

Compress a single chain: the SVD route canonicalizes a copy of the input with a
truncating SVD sweep; the variational (ALS) route draws a `svdguess_compress(x, D)`
initial guess and refines it by sweeps, returning `(chain, info)` with `info`, the
[`ALSConvergenceInfo`](@ref) of the sweeps (the SVD route returns the chain only).
"""
function compress(ψ::CanonicalMPS, alg::SVDCompression=DefaultMultAlg)
	return _canonicalize!(copy(ψ); alg=Orthogonalize(SVD(), alg.trunc, false))[1]
end
function compress(h::AbstractMPO, alg::SVDCompression=DefaultMultAlg)
	out, _ = _canonicalize!(copy(h); alg=Orthogonalize(SVD(), alg.trunc, false))
	return CanonicalMPO(out.data)
end
# block-sparse chains: operator gauging is written on 4-index site tensors
compress(h::MPOHamiltonian, alg::SVDCompression=DefaultMultAlg) =
	compress(MPO(tompotensors(h)), alg)

function compress(x, alg::DMRG1)
	out = svdguess_compress(x, alg.D)
	return compress!(out, x, alg)
end
compress(h::MPOHamiltonian, alg::DMRG1) =
	compress(MPO(tompotensors(h)), alg)

# ---------- two-site (DMRG2) sweeps and interface ----------

# the two-site optimal overlap block from the fused left tensor K (= cL · ket[s], the same
# `_fuse_left` the single-site sweeps use): one contraction with the fused right site
# (ket[s+1] into the right overlap environment)
function _map_fused2(K::AbstractArray{T,3}, c::OverlapCache, s::Integer) where {T}
	return @tensor t[bL, -2, -3, -4] := K[bL, -2, km] * c.ket[s+1][km, -3, kR] *
									   c.cstorage[s+2][-4, kR]
end
function _map_fused2(K::AbstractArray{T,4}, c::OverlapCache, s::Integer) where {T}
	return @tensor t[b, -2, -3, -4, -5, xR] := K[b, -2, m, -4] * c.ket[s+1][m, -3, kR, -5] *
											  c.cstorage[s+2][xR, kR]
end

# the mirror image: the fused right tensor S (= ket[s+1] · cR) and the left site chain
# (cL · ket[s])
function _map_fused2R(S::AbstractArray{T,3}, c::OverlapCache, s::Integer) where {T}
	return @tensor t[-1, -2, -3, -4] := c.cstorage[s][-1, kL] * c.ket[s][kL, -2, km] *
									   S[km, -3, -4]
end
function _map_fused2R(S::AbstractArray{T,4}, c::OverlapCache, s::Integer) where {T}
	return @tensor t[b, -2, -3, -4, -5, xR] := c.cstorage[s][b, kL] * c.ket[s][kL, -2, m, -4] *
											  S[m, -3, xR, -5]
end

function leftsweep!(c::OverlapCache, alg::DMRG2)
	L = length(c.bra)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		K = _fuse_left(c, s)
		t2 = _map_fused2(K, c, s)
		kvals[s] = norm(t2)
		(alg.verbosity > 2) && _logupdate(stdout, "l2r", s, kvals[s])
		_als2_update!(c.bra, s, t2, alg; move_right=true)
		_push_left_fused!(c, K, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

function rightsweep!(c::OverlapCache, alg::DMRG2)
	L = length(c.bra)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		S = _fuse_right(c, s)
		t2 = _map_fused2R(S, c, s - 1)
		kvals[k] = norm(t2)
		(alg.verbosity > 2) && _logupdate(stdout, "r2l", s - 1, kvals[k])
		k += 1
		_als2_update!(c.bra, s - 1, t2, alg; move_right=false)
		_push_right_fused!(c, S, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

sweep!(c::OverlapCache, alg::DMRG2) = vcat(leftsweep!(c, alg), rightsweep!(c, alg))

function compress!(out, x, alg::DMRG2)
	cache = OverlapCache(out, x)
	info = iterative_compute!(cache, alg)
	# attach the external operand scale and fold the center norm into `scaling` (see `mult!`)
	x isa Union{CanonicalMPS, CanonicalMPO} && setscaling!(out, scaling(x))
	_renormalize!(out, out[1], false)
	return out, info
end

function compress(x, alg::DMRG2)
	out = svdguess_compress(x, _guess_bond(alg.trunc))
	return compress!(out, x, alg)
end
compress(h::MPOHamiltonian, alg::DMRG2) = compress(MPO(tompotensors(h)), alg)
