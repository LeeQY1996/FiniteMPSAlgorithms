# TEBD: quantum gates (AbstractGate / GateOp), their application (apply!) and site
# swaps (swap!), gate fusion (fuse_gates) and the Trotter propagator circuit builder
# (trotter_propagator). Minimal TEBD building blocks; the time-evolution loop is
# driven by the caller.

"""
	AbstractGate{N,T}

Abstract supertype of quantum gates acting on `N` sites: `positions(g)::NTuple{N,Int}`
(ascending), `g.op::Array{T,M}` — the rank-`M = 2N` operator tensor in the index
convention `(i1', i2', …, iN', i1, …, iN)`, i.e. all bra (output) indices first and all
ket (input) indices second, each block ordered with site 1 the slowest index.
"""
abstract type AbstractGate{N, T} end

positions(g::AbstractGate) = g.positions
scalartype(::Type{<:AbstractGate{N, T}}) where {N, T} = T

"""
	shift(g, by)

Shift all support positions of gate `g` by `by` sites.
"""
shift(g::G, by::Integer) where {G<:AbstractGate} =
	typeof(g)(ntuple(i -> g.positions[i] + by, Val(length(g.positions))), g.op)

function Base.adjoint(g::G) where {G<:AbstractGate}
	N = length(g.positions)
	# U† : swap the ket/bra index blocks and conjugate; support positions unchanged
	perm = (ntuple(i -> N + i, Val(N))..., ntuple(i -> i, Val(N))...)
	return typeof(g)(g.positions, permutedims(conj(g.op), perm))
end

"""
	GateOp(positions, op)

A quantum gate acting on the `N` distinct sites `positions`. The operator `op` is a
rank-`2N` tensor `(i1', …, iN', i1, …, iN)` — all bra (output) indices first, all
ket (input) indices second, each block ordered with `positions[1]` the slowest index.
The positions only have to be distinct: unsorted inputs are sorted internally (the
operator legs follow along). The input is copied and materialized as a dense array of
concrete element type; it is *not* checked for unitarity (pass `exp(-im * dt * h)` for
a unitary gate). Since a non-unitary gate does not preserve the canonical form under
the Hastings update, [`apply!`](@ref) can re-canonicalize the state with `truncate!`
(`canonicalize = true`).
"""
struct GateOp{N, T, M} <: AbstractGate{N, T}
	positions::NTuple{N, Int}
	op::Array{T, M}   # (i1', …, iN', i1, …, iN) with M == 2N

	function GateOp{N, T, M}(positions::NTuple{N, Int}, op::AbstractArray) where {N, T, M}
		M == 2N || throw(ArgumentError("the operator tensor must have rank $(2N)"))
		ndims(op) == M || throw(DimensionMismatch("the operator tensor must have rank $M"))
		allunique(positions) || throw(ArgumentError("the gate positions must be distinct"))
		return new{N, T, M}(positions, convert(Array{T, M}, op))
	end
end

function GateOp(positions::NTuple{N, Int}, op::AbstractArray) where {N}
	M = ndims(op)
	M == 2N || throw(ArgumentError("the operator tensor must have rank $(2N)"))
	T = scalartype(op)
	perm = sortperm(collect(positions))
	if perm != collect(1:N)
		# sort the support and follow with the operator legs: sorted site j is input
		# site perm[j]
		op = permutedims(op, (perm..., (N .+ perm)...))
		positions = ntuple(i -> positions[perm[i]], Val(N))
	end
	return GateOp{N, T, M}(positions, op)
end

"""
	isunitary(g::GateOp; atol=1e-10) -> Bool

Whether the gate operator of `g` is unitary: the fused `(d^N × d^N)` operator matrix
satisfies `U†·U ≈ I` within `atol`.
"""
function isunitary(g::GateOp{N}; atol::Real=1.0e-10) where {N}
	m = tie(g.op, (N, N))
	return isapprox(m' * m, isometry(scalartype(g), size(m, 2)); atol)
end

# ── N-site gate application (Hastings update, aligned with TEMPO/GTEMPO; the
#    multi-site block sweep follows MPSSimulator) ─────────────────────────────────

"contract the contiguous block `a … a+N-1` of right-canonical factors into the plain
block tensor `(Dl, P, Dr)`; the fused physical index `P` runs with site `a` the fastest
digit."
function _contract_block(ψ::CanonicalMPS, a::Integer, N::Integer)
	t = ψ[a]
	Dl = size(t, 1)
	for k in 1:N-1
		A = ψ[a+k]
		t = reshape(t, :, size(t, 3)) * reshape(A, size(A, 1), :)
		t = reshape(t, Dl, :, size(A, 3))
	end
	return t
end

"""
Apply the `N`-site gate `g` to the contiguous block `a … a+N-1` of `ψ` (in place,
Hastings update). The block is contracted into the unweighted tensor `t`, the gate
acts on its fused physical legs, and the block is re-decomposed bond by bond: each
step SVDs the spectrum-weighted block, stores the new spectrum, writes the right
factor, and recovers the left site by back-projecting the unweighted block through
`v†` (Hastings), so no spectrum is ever inverted.
"""
function _gate_block_apply!(g::AbstractGate{N}, ψ::CanonicalMPS, a::Integer;
							trunc::TruncationScheme) where {N}
	(1 <= a && a + N - 1 <= length(ψ)) || throw(BoundsError())
	dims = [phydim(ψ[j]) for j in a:(a+N-1)]   # physical dims, captured pre-overwrite
	svectors_uninitialized(ψ) && canonicalize!(ψ)
	t = _contract_block(ψ, a, N)               # (Dl, P, Dr); site a the fastest digit
	dn = size(t, 2)
	# gate matrix on the fused physical legs: the column-major reshape of the operator
	# tensor and of the block's fused physical index share the same little-endian site
	# order (support site 1 the least significant digit), rows = outputs, cols = inputs
	G = reshape(g.op, dn, dn)
	t = permutedims(t, (2, 1, 3))              # (P, Dl, Dr)
	t = reshape(G * reshape(t, dn, :), dn, size(t, 2), size(t, 3))
	t = permutedims(t, (2, 1, 3))              # (Dl, P, Dr), post-gate and unweighted
	Dr = size(t, 3)
	Dl = size(t, 1)
	Λ = ψ.s[a]                                 # left-bond spectrum (preconditioner)
	for k in 1:N-1
		d = dims[k]
		# weighted block at bond a+k-1, grouped (l, p) | (rest…)
		u, s, v, = tsvd!(reshape(Λ .* t, Dl * d, :); trunc)
		# Hastings: the left site is the unweighted block back-projected through v†
		ψ[a+k-1] = reshape(reshape(t, Dl * d, :) * adjoint(v), Dl, d, length(s))
		ψ.s[a+k] = s
		t = reshape(copy(v), length(s), :, Dr) # the (remaining) weighted block
		Λ = s
		Dl = length(s)
	end
	ψ[a+N-1] = reshape(t, Dl, dims[N], Dr)
	return ψ
end

"""
	apply!(g::AbstractGate, ψ::CanonicalMPS; trunc=Defaults.alg_trunc(),
	       canonicalize=false) -> ψ

Apply the gate `g` (acting on `N ≥ 1` distinct sites) to `ψ` in place, with the
Hastings update (aligned with TEMPO/GTEMPO; the multi-site block sweep after
MPSSimulator). The gate sites may be any distinct set of positions: non-adjacent
supports are first gathered into a contiguous block with exact content swaps
([`swap!`](@ref)), the gate is applied on the block bond by bond, and the gathering
swaps are undone. Initializes the canonical form if needed.

**Mathematics of the Hastings update.** In the right-canonical gauge the left
environment at the current bond `b` is `diag(ψ.s[b]²)`. With the unweighted post-gate
block tensor `t` and the bond spectrum `Λ`, the wavefunction block in the left Schmidt
basis is the *folded* tensor `W = diag(Λ) · t`. Decompose it,

	W = u · diag(σ) · v,		u'·u = I,	v·v' = I,

(with truncation, `σ` keeps the leading singular values and `err` is the discarded
weight), and assign — moving left to right through the block —

	ψ[b] = t · v†,		ψ.s[b+1] = σ,		t ← v.

Three properties make this exact up to the discarded weights, without ever inverting a
spectrum:

* *state*: the block `t` is reproduced, because `v†·v` is the projector onto the row
  space of `W`, and the rows of `t` lie in it;
* *gauge*: for unitary gates the back-projected left factor `t · v†` is
  right-orthogonal — `t†·t = I`, since `t` contracts right-canonical factors and the
  gate drops out (`G†·G = I`);
* *spectrum*: the new left environment after the bond is
  `v · (t† · diag(Λ²) · t) · v† = v · (W†·W) · v† = diag(σ²)`.

If the input is right-canonical and the truncation error is small, the right-canonical
form is preserved without any `canonicalize!`. The fold–SVD–back-project step is the
two-site application trick introduced for the light-cone TEBD by M. Hastings, "Light
cone Matrix Product", *J. Math. Phys.* **50**, 095207 (2009) (aligned with
TEMPO/GTEMPO); its advantage is that both sides of every bond are updated from a
*single* decomposition while keeping the gauge and the local spectrum consistent.
**Caveat:** the exact preservation of the strict canonical form relies on the
truncated SVD being exact for the state at hand — for aggressive truncations the
*gauge* property degrades and the canonical form is only approximate. A non-unitary
gate breaks the *gauge* property outright; pass `canonicalize = true` to
re-canonicalize the final state with [`truncate!`](@ref) (an SVD sweep under `trunc`).
"""
function apply!(g::AbstractGate{N}, ψ::CanonicalMPS; trunc::TruncationScheme=Defaults.alg_trunc(),
				canonicalize::Bool=false) where {N}
	L = length(ψ)
	N <= L || throw(BoundsError())
	all(p -> 1 <= p <= L, positions(g)) || throw(BoundsError())
	if N == 1
		_gate_block_apply!(g, ψ, positions(g)[1]; trunc)
		canonicalize && truncate!(ψ; trunc)
		return ψ
	end
	# gather the supports into the contiguous block pos[1] … pos[1]+N-1
	a = positions(g)[1]
	swaps = Pair{Int,Int}[]
	for k in 2:N
		s = positions(g)[k]
		t = a + k - 1
		while s > t
			swap!(ψ, s - 1; trunc)
			push!(swaps, s - 1 => s)
			s -= 1
		end
	end
	_gate_block_apply!(g, ψ, a; trunc)
	canonicalize && truncate!(ψ; trunc)
	# undo the gathering swaps (in inverse order)
	for (i, _) in reverse(swaps)
		swap!(ψ, i; trunc)
	end
	return ψ
end

"""
	apply!(gs, ψ::CanonicalMPS; kwargs...) -> ψ

Apply a sequence of gates (a circuit) in order — [`apply!`](@ref) gate by gate.
"""
apply!(gs::AbstractVector{<:AbstractGate}, ψ::CanonicalMPS; kwargs...) =
	(foreach(g -> apply!(g, ψ; kwargs...), gs); ψ)

"""
	swap!(ψ::CanonicalMPS, i::Integer; trunc=Defaults.alg_trunc()) -> ψ

Exchange the physical content of the neighboring sites `i` and `i+1` of `ψ` using the
Hastings SWAP gate (aligned with TEMPO/GTEMPO): the two-site block is formed with the
physical legs crossed, the Schmidt spectrum `psi.s[i]` on the left bond is contracted
into it, and the crossed block is re-decomposed with truncation. The right factor is
written directly to `psi[i+1]`, the renewed spectrum to `psi.s[i+1]`, and the left
site is recovered by back-projecting the crossed block. If the input is
right-canonical and the truncation error is small, the right-canonical form is
preserved without any `canonicalize!` (the SWAP gate is unitary). Sequences of
`swap!` (`permutation2swaps`) generate arbitrary permutations of the site contents.
Initializes the canonical form if needed.
"""
function swap!(ψ::CanonicalMPS, i::Integer; trunc::TruncationScheme=Defaults.alg_trunc())
	(1 <= i <= length(ψ) - 1) || throw(BoundsError())
	svectors_uninitialized(ψ) && canonicalize!(ψ)
	A = ψ[i]
	B = ψ[i+1]
	# two-site block with the physical legs crossed: the leg at site i carries the
	# physical content of B, the leg at site i+1 that of A
	@tensor block[a, q, p, c] := A[a, p, b] * B[b, q, c]
	# Hastings: fold the Schmidt spectrum on the left bond into the block, then SVD
	sv = Diagonal(ψ.s[i])
	@tensor weighted[a, q, p, c] := sv[a, 1] * block[1, q, p, c]
	u, s, v, err = tsvd!(weighted, (1, 2), (3, 4); trunc)
	ψ[i+1] = v
	ψ.s[i+1] = s
	# back-project the crossed block; the left site keeps the (crossed) physical leg of B
	@tensor Anew[a, q, c] := block[a, q, p, c'] * conj(v[c, p, c'])
	ψ[i] = Anew
	return ψ
end

"""
	swap!(ρ::CanonicalMPO, i::Integer; trunc=Defaults.alg_trunc()) -> ρ

Exchange the physical content `(p_out, p_in)` of the neighboring sites `i` and `i+1`
of the operator chain `ρ` — the operator-space analog of the MPS [`swap!`](@ref)
(Hastings update, aligned with TEMPO/GTEMPO): the two-site block is formed with the
physical leg pairs crossed, the Schmidt spectrum `rho.s[i]` on the left bond is
contracted into it, and the crossed block is re-decomposed with truncation. The right
factor is written directly to `rho[i+1]`, the renewed spectrum to `rho.s[i+1]`, and
the left site is recovered by back-projecting the crossed block. If the input is
right-canonical and the truncation error is small, the right-canonical form is
preserved without any `canonicalize!`. Sequences of `swap!` (`permutation2swaps`)
generate arbitrary permutations of the site contents. Initializes the canonical form
if needed.
"""
function swap!(ρ::CanonicalMPO, i::Integer; trunc::TruncationScheme=Defaults.alg_trunc())
	(1 <= i <= length(ρ) - 1) || throw(BoundsError())
	svectors_uninitialized(ρ) && canonicalize!(ρ)
	W = ρ[i]
	V = ρ[i+1]
	# two-site block with the physical leg pairs crossed: the legs at site i carry the
	# (p_out, p_in) content of V, the legs at site i+1 that of W
	@tensor block[a, qo, qi, po, pin, b] := W[a, po, m, pin] * V[m, qo, b, qi]
	# Hastings: fold the Schmidt spectrum on the left bond into the block, then SVD
	sv = Diagonal(ρ.s[i])
	@tensor weighted[a, qo, qi, po, pin, b] := sv[a, 1] * block[1, qo, qi, po, pin, b]
	u, s, v, err = tsvd!(weighted, (1, 2, 3), (4, 5, 6); trunc)
	ρ[i+1] = permutedims(v, (1, 2, 4, 3))          # (aL, po, aR, pin)
	ρ.s[i+1] = s
	# back-project the crossed block; the left site keeps the (crossed) legs of V
	@tensor Wnew[a, qo, m, qi] := block[a, qo, qi, po, pin, b] * conj(v[m, po, pin, b])
	ρ[i] = Wnew
	return ρ
end

# ── gate fusion (aligned with QuantumSpins.fuse_gates) ──────────────────────────
# two gates can merge into one iff their supports intersect (gates with disjoint
# supports commute); the merged support is the union, with identities embedded on the
# complementary sites. Merges that would exceed `maxsites` sites are skipped.

"gates with disjoint supports commute."
_gates_commute(a::NTuple{N, Int}, b::NTuple{M, Int}) where {N, M} = !any(in(b), a)

"embed the gate operator into the (superset) support `pos` and return the fused `(D,
D)` matrix (site 1 the slowest index); the local dimensions of the sites missing from
the gate's own support are taken from `other`, which covers them."
function _gate_embed(g::AbstractGate{N, T}, pos::Vector{Int}, other::AbstractGate) where {N, T}
	ka = collect(g.positions)
	D = prod(ntuple(i -> size(g.op, i), Val(N)))
	ka == pos && return reshape(g.op, D, D)
	po = collect(other.positions)
	# complementary sites (ascending) and their local dimensions from `other`
	ms = Int[p for p in pos if p ∉ ka]
	dms = [size(other.op, findfirst(==(p), po)) for p in ms]
	n = length(pos)
	# step 1: identity padding on the complementary sites; legs (ka…, ms…, ka…, ms…)
	# (kron puts the LEFT factor's row index in the high digits, i.e. the first legs)
	δ = reshape(Matrix{T}(I, prod(dms), prod(dms)), dms..., dms...)
	E = kron(δ, reshape(g.op, D, D))
	# step 2: reorder the legs from (ka…, ms…) to the ascending `pos`
	perm = ntuple(Val(2n)) do l
		j = l <= n ? l : l - n
		i = findfirst(==(pos[j]), ka)
		isnothing(i) && (i = N + findfirst(==(pos[j]), ms))
		return l <= n ? i : n + i
	end
	dims = (ntuple(i -> size(g.op, i), Val(N))..., dms...,
			ntuple(i -> size(g.op, i), Val(N))..., dms...)
	return reshape(permutedims(reshape(E, dims...), perm), D * prod(dms), D * prod(dms))
end

"merge the two gates as `a ∘ b` (`b` applied first) on the union support; returns
`nothing` when the supports are disjoint or the union exceeds `maxsites` sites."
function _try_gate_mul(a::AbstractGate, b::AbstractGate, maxsites::Int)
	na, nb = length(positions(a)), length(positions(b))
	pos = sort(union(positions(a), positions(b)))
	length(pos) == na + nb && return nothing      # disjoint supports (commuting gates)
	length(pos) <= max(maxsites, max(na, nb)) || return nothing
	T = promote_type(scalartype(a), scalartype(b))
	# `a` acts after `b`
	r = convert(Matrix{T}, _gate_embed(a, pos, b)) * convert(Matrix{T}, _gate_embed(b, pos, a))
	dims = map(pos) do p
		i = findfirst(==(p), collect(positions(a)))
		isnothing(i) ? size(b.op, findfirst(==(p), collect(positions(b)))) : size(a.op, i)
	end
	return GateOp(Tuple(pos), reshape(r, dims..., dims...))
end

"one fusion pass: every gate is absorbed backwards into the closest earlier gate with
an intersecting support (the scan stops at gates it does not commute past)."
function _fuse_gates_pass(gs::Vector{<:AbstractGate}, maxsites::Int)
	b = AbstractGate[gs...]   # heterogeneous storage: merged supports can grow
	for i in lastindex(b):-1:2
		item = b[i]
		br = b[1:i-1]
		for j in lastindex(br):-1:1
			r = _try_gate_mul(item, br[j], maxsites)
			if r !== nothing
				br[j] = r
				b = [br; b[i+1:end]]
				break
			end
			_gates_commute(positions(item), positions(br[j])) || break
		end
	end
	return b
end

# the adjoint of a gate sequence: reversed order, every gate adjointed
_fuse_adjoints(gs::AbstractVector{<:AbstractGate}) = [adjoint(g) for g in reverse(gs)]

"""
	fuse_gates(gates::AbstractVector{<:AbstractGate}; maxsites=4) -> Vector{GateOp}

Fuse a gate sequence (circuit) into as few gates as possible (aligned with
QuantumSpins): every gate is absorbed backwards into the closest earlier gate with an
intersecting support — two gates commute exactly when their supports are disjoint —
and a second pass on the adjointed circuit (a forward pass) merges the boundary gates
of repeated layers. Gates whose union support would exceed `maxsites` sites are left
unfused. The fused circuit applies the same product of operators as the input.
"""
function fuse_gates(gates::AbstractVector{<:AbstractGate}; maxsites::Int=4)
	return _fuse_adjoints(_fuse_gates_pass(_fuse_adjoints(collect(gates)), maxsites))
end

# ── Trotter propagator (aligned with QuantumSpins.trotter_propagator; algorithm
#    reference: "Higher order decompositions of ordered operator exponentials") ───

"absorb the one-site terms of `ham` into nearest-neighbor terms (`h ⊗ I` / `I ⊗ h`)."
function _absorb_one_bodies(ham::OpSum)
	L = length(ham.ds)
	(L >= 2) || throw(ArgumentError("the lattice needs at least two sites"))
	T = scalartype(ham)
	terms = OpTerm{T}[]
	for t in ham.data
		if length(t.positions) == 1
			i = t.positions[1]
			O = t.operators[1]
			if i < L
				push!(terms, OpTerm(t.coeff, [i, i + 1],
									[O, Matrix{T}(I, ham.ds[i+1], ham.ds[i+1])]))
			else
				push!(terms, OpTerm(t.coeff, [i - 1, i],
									[Matrix{T}(I, ham.ds[i-1], ham.ds[i-1]), O]))
			end
		else
			push!(terms, t)
		end
	end
	return OpSum(ham.ds, terms)
end

"one gate per distinct term support: the terms of a support are summed into
`H` and the gate realizes `exp(τ · H)` on that support (sorted by position). The terms
must carry constant coefficients (freeze them with [`_freeze_term`](@ref) first)."
function _expm_terms(ts::Vector{<:OpTerm}, τ::Number)
	groups = Dict{Vector{Int},Vector{OpTerm}}()
	for t in ts
		push!(get!(groups, t.positions, Vector{OpTerm}()), t)
	end
	gs = GateOp[]
	for pos in sort!(collect(keys(groups)))
		tsk = groups[pos]
		H = tsk[1].coeff.f .* kron(tsk[1].operators...)
		for t in tsk[2:end]
			H .+= t.coeff.f .* kron(t.operators...)
		end
		n = length(pos)
		dims = ntuple(i -> size(tsk[1].operators[i], 1), Val(n))
		# the column-major reshape makes the LAST support site the fastest digit; flip
		# to the gate convention (support site 1 the first leg / slowest index)
		perm = (ntuple(i -> n + 1 - i, Val(n))..., ntuple(i -> 2n + 1 - i, Val(n))...)
		push!(gs, GateOp(Tuple(pos), permutedims(reshape(exp(Matrix(τ * H)), dims..., dims...), perm)))
	end
	return gs
end

"split a nearest-neighbor two-site term list into the even and odd bond families."
function _split_nn(ts)
	even = Vector{OpTerm}()
	odd = Vector{OpTerm}()
	for t in ts
		(length(t.positions) == 2 && t.positions[2] == t.positions[1] + 1) ||
			throw(ArgumentError("the even/odd splitting requires nearest-neighbor two-site terms"))
		t.positions[1] % 2 == 0 ? push!(even, t) : push!(odd, t)
	end
	return even, odd
end

"second-order Strang layer for nearest-neighbor hamiltonians: even bonds at `τ/2`,
odd bonds at `τ`, even bonds at `τ/2`."
function _trotter_layer_nn(ts::Vector{<:OpTerm}, τ::Number)
	even, odd = _split_nn(ts)
	return vcat(_expm_terms(even, τ / 2), _expm_terms(odd, τ), _expm_terms(even, τ / 2))
end

"generic second-order layer for arbitrary supports: the support gates at `τ/2`, then
reversed."
function _trotter_layer_generic(ts::Vector{<:OpTerm}, τ::Number)
	gs = _expm_terms(ts, τ / 2)
	return vcat(gs, reverse(gs))
end

"freeze a term's coefficient at time `tval` (constants pass through)."
_freeze_term(t::OpTerm, tval::Number) =
	isconstant(t) ? t : OpTerm(t.coeff(tval), t.positions, t.operators)

"one Trotter layer of half-order `p` over the interval starting at `ti` with duration
`τ`: `p == 1` is the second-order Strang splitting (the coefficients are frozen at the
interval midpoint for a time-dependent `ham`), higher orders use the recursive Suzuki
fractal with fraction `sp = 1 / (4 - 4^(1/(2p-1)))` over the five sub-intervals."
function _trotter_layer(ts::Vector{<:OpTerm}, ti::Number, τ::Number, p::Integer;
						constant::Bool)
	if p == 1
		tsv = constant ? ts : [_freeze_term(t, ti + τ / 2) for t in ts]
		all(t -> length(t.positions) == 2 && t.positions[2] == t.positions[1] + 1, tsv) &&
			return _trotter_layer_nn(tsv, τ)
		return _trotter_layer_generic(tsv, τ)
	end
	sp = 1 / (4 - 4^(1 / (2p - 1)))
	return vcat(_trotter_layer(ts, ti, sp * τ, p - 1; constant),
				_trotter_layer(ts, ti + sp * τ, sp * τ, p - 1; constant),
				_trotter_layer(ts, ti + 2sp * τ, (1 - 4sp) * τ, p - 1; constant),
				_trotter_layer(ts, ti + (1 - 2sp) * τ, sp * τ, p - 1; constant),
				_trotter_layer(ts, ti + (1 - sp) * τ, sp * τ, p - 1; constant))
end

"""
	trotter_propagator(ham::OpSum, tspan=(t₀, t₁); order=2, stepsize=0.01) -> Vector{GateOp}

Build the gate sequence (circuit) of the Trotter-Suzuki propagator `exp(τ · ham)` over
`tspan` in steps of length `stepsize` (aligned with QuantumSpins; algorithm reference:
"Higher order decompositions of ordered operator exponentials"). Only even `order` is
supported: `order = 2` is the Strang splitting (a purely nearest-neighbor `ham` uses
the even/odd bond splitting; the generic case builds one gate per distinct term
support), higher orders recurse into the Suzuki fractal of five half-order layers.
One-site terms are absorbed into nearest-neighbor terms (`h ⊗ I`).

Time-dependent coefficients ([`Coeff`](@ref) functions) are supported: every (sub-)
layer freezes the coefficients at its own midpoint. The gates realize the bare
exponential `exp(τ · H)` per support — pass `tspan = (0, -im * t)` for the real-time
propagator `exp(-im · t · H)`. The circuit can be compressed with [`fuse_gates`](@ref)
and applied with [`apply!`](@ref apply!(gs, ψ::CanonicalMPS; kwargs...)).
"""
function trotter_propagator(ham::OpSum, tspan::Tuple{<:Number,<:Number};
							order::Int=2, stepsize::Number=0.01)
	p = div(order, 2)
	(p * 2 == order) || throw(ArgumentError("only even orders are supported, got order = $order"))
	ts = _absorb_one_bodies(ham).data
	constant = isconstant(ham)
	ti, tf = tspan
	τ = tf - ti
	nsteps = max(round(Int, abs(τ / stepsize)), 1)
	dτ = τ / nsteps
	circuit = GateOp[]
	for i in 1:nsteps
		append!(circuit, _trotter_layer(ts, ti + (i - 1) * dτ, dτ, p; constant))
	end
	return circuit
end
