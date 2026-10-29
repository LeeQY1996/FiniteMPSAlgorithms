# Trotter circuit construction and gate fusion. Gate and swap primitives live in MPSCore.

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
	# (kron puts the LEFT factor's row index in the high digits, i.e. the last legs)
	δ = Matrix{T}(I, prod(dms), prod(dms))
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
`nothing` when the supports are disjoint."
function _try_gate_mul(a::AbstractGate, b::AbstractGate)
	na, nb = length(positions(a)), length(positions(b))
	pos = sort(union(positions(a), positions(b)))
	length(pos) == na + nb && return nothing      # disjoint supports (commuting gates)
	T = promote_type(scalartype(a), scalartype(b))
	# `a` acts after `b`
	r = convert(Matrix{T}, _gate_embed(a, pos, b)) * convert(Matrix{T}, _gate_embed(b, pos, a))
	dims = map(pos) do p
		i = findfirst(==(p), collect(positions(a)))
		isnothing(i) ? size(b.op, findfirst(==(p), collect(positions(b)))) : size(a.op, i)
	end
	# identity embeddings and operator products preserve unitarity
	return Gate(Tuple(pos), reshape(r, dims..., dims...);
				  isunitary=a.isunitary && b.isunitary)
end

"one fusion pass: every gate is absorbed backwards into the closest earlier gate with
an intersecting support (the scan stops at gates it does not commute past)."
function _fuse_gates_pass(gs::Vector{<:AbstractGate})
	b = AbstractGate[gs...]   # heterogeneous storage: merged supports can grow
	for i in lastindex(b):-1:2
		item = b[i]
		br = b[1:i-1]
		for j in lastindex(br):-1:1
			r = _try_gate_mul(item, br[j])
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
	fuse_gates(gates::AbstractVector{<:AbstractGate}) -> Vector{Gate}

Fuse a gate sequence (circuit) into as few gates as possible (aligned with
QuantumSpins): every gate is absorbed backwards into the closest earlier gate with an
intersecting support — two gates commute exactly when their supports are disjoint —
and a second pass on the adjointed circuit (a forward pass) merges the boundary gates
of repeated layers. Single-site gates are absorbed into any intersecting multi-site
gate, so a gate whose support covers a site swallows that site's single-site gates. No
gate is left out: every intersecting pair is mergeable, and the fused circuit applies
the same product of operators as the input.
"""
fuse_gates(gates::AbstractVector{<:AbstractGate}) =
	_fuse_adjoints(_fuse_gates_pass(_fuse_adjoints(collect(gates))))

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
function _expm_terms(ts::Vector{<:OpTerm}, τ::Real)
	groups = Dict{Vector{Int},Vector{OpTerm}}()
	for t in ts
		push!(get!(groups, t.positions, Vector{OpTerm}()), t)
	end
	gs = Gate[]
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
		push!(gs, Gate(Tuple(pos), permutedims(reshape(exp(Matrix(τ * H)), dims..., dims...), perm)))
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
function _trotter_layer_nn(ts::Vector{<:OpTerm}, τ::Real)
	even, odd = _split_nn(ts)
	return vcat(_expm_terms(even, τ / 2), _expm_terms(odd, τ), _expm_terms(even, τ / 2))
end

"generic second-order layer for arbitrary supports: the support gates at `τ/2`, then
reversed."
function _trotter_layer_generic(ts::Vector{<:OpTerm}, τ::Real)
	gs = _expm_terms(ts, τ / 2)
	return vcat(gs, reverse(gs))
end

"freeze a term's coefficient at time `tval` (constants pass through)."
_freeze_term(t::OpTerm, tval::Real) =
	isconstant(t) ? t : OpTerm(t.coeff(tval), t.positions, t.operators)

"one Trotter layer of half-order `p` over the interval starting at `ti` with duration
`τ`: `p == 1` is the second-order Strang splitting (the coefficients are frozen at the
interval midpoint for a time-dependent `ham`), higher orders use the recursive Suzuki
fractal with fraction `sp = 1 / (4 - 4^(1/(2p-1)))` over the five sub-intervals."
function _trotter_layer(ts::Vector{<:OpTerm}, ti::Real, τ::Real, p::Integer;
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
	trotter_propagator(ham::OpSum, tspan::Tuple{<:Real,<:Real};
					   order=2, stepsize=0.01) -> Vector{Gate}

Build the gate sequence (circuit) of the Trotter-Suzuki propagator `exp(τ · ham)` over
the real interval `tspan` in steps of length `stepsize` (aligned with QuantumSpins;
algorithm reference: "Higher order decompositions of ordered operator exponentials").
Only even `order` is supported: `order = 2` is the Strang splitting (a purely
nearest-neighbor `ham` uses the even/odd bond splitting; the generic case builds one
gate per distinct term support), higher orders recurse into the Suzuki fractal of five
half-order layers. One-site terms are absorbed into nearest-neighbor terms
(`h ⊗ I`).

Time-dependent coefficients ([`Coeff`](@ref) functions) are supported: every (sub-)
layer freezes the coefficients at its own (real) midpoint.

The circuit realizes the *bare* exponential `exp(τ · H)` of the hamiltonian that is
passed in: the gate on a support is `exp(τ · H_support)`, with `τ` real. Hence the
physical propagators are obtained by scaling the hamiltonian, while `tspan` stays
real:

- imaginary-time evolution `exp(-β · H)`: pass `-ham` with `tspan = (0, β)`;
- real-time evolution `exp(-i · t · H)`: pass `-im * ham` with `tspan = (0, t)`.

Passing `ham` unscaled yields `exp(+τ · H)`, i.e. the imaginary-time propagator of
`-ham`, not `exp(-i t H)`. A complex `tspan` is rejected (the time must be real).
The circuit can be compressed with [`fuse_gates`](@ref) and applied with
[`apply!`](@ref apply!(gs, ψ::CanonicalMPS; kwargs...)).
"""
function trotter_propagator(ham::OpSum, tspan::Tuple{<:Real,<:Real};
							order::Int=2, stepsize::Real=0.01)
	p = div(order, 2)
	(p * 2 == order) || throw(ArgumentError("only even orders are supported, got order = $order"))
	ts = _absorb_one_bodies(ham).data
	constant = isconstant(ham)
	ti, tf = tspan
	τ = tf - ti
	nsteps = max(round(Int, abs(τ / stepsize)), 1)
	dτ = τ / nsteps
	circuit = Gate[]
	for i in 1:nsteps
		append!(circuit, _trotter_layer(ts, ti + (i - 1) * dτ, dτ, p; constant))
	end
	return circuit
end
