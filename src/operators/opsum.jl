# OpTerm: a product operator term with support on specific sites.
# OpSum: a validated collection of OpTerms on a shared lattice, built by `+`.

"""
	Coeff(x)
	Coeff(f)

A scalar coefficient of an [`OpTerm`](@ref): a plain number `x`, or a time-dependent
function `f` mapping a real time to a number (a bare function is wrapped as `Coeff(f)`).
`(c::Coeff)(t::Real)` evaluates the coefficient at the real time `t` (a constant
ignores `t`), [`isconstant`](@ref) says whether it is time-independent, and
`scalartype(c)` probes the produced number type (a function coefficient is probed at
`t = 0.0`). Multiplying a coefficient by a number scales it pointwise.
Time-dependent coefficients are only supported by [`trotter_propagator`](@ref).
"""
struct Coeff{F}
	f::F
end
Coeff(x::Number) = Coeff{typeof(x)}(x)

(c::Coeff)(t::Real) = c.f isa Number ? c.f : c.f(t)
isconstant(c::Coeff) = c.f isa Number
scalartype(c::Coeff) = c.f isa Number ? typeof(c.f) : typeof(c.f(0.0))
Base.:(==)(a::Coeff, b::Coeff) = a.f == b.f
Base.:(==)(a::Coeff, x::Number) = isconstant(a) && a.f == x
Base.:(==)(x::Number, a::Coeff) = a == x
Base.conj(c::Coeff) = isconstant(c) ? Coeff(conj(c.f)) : Coeff(t -> conj(c.f(t)))
# the hermitian conjugate of a coefficient (a time-dependent one is conjugated pointwise)
_adjoint(c::Coeff) = isconstant(c) ? Coeff(conj(c.f)) : Coeff(t -> conj(c.f(t)))
# scalar multiplication (a time-dependent coefficient is scaled pointwise)
Base.:*(x::Number, c::Coeff) = isconstant(c) ? Coeff(x * c.f) : Coeff(t -> x * c.f(t))
Base.:*(c::Coeff, x::Number) = x * c
Base.:-(c::Coeff) = -1 * c

"""
	OpTerm(coeff, positions, operators)
	OpTerm(coeff, i₁ => O₁, i₂ => O₂, ...)

A product operator term `coeff * O₁ ⊗ O₂ ⋯` acting on the site positions `positions`
(each position at most once; the term is symmetrized in any order, since the operators
commute on distinct sites). The coefficient may be a number, a time-dependent function
`t -> x(t)` or a [`Coeff`](@ref) — time-dependent terms are only supported by
[`trotter_propagator`](@ref). The operators only have to be *square*: the local dimension
may differ from site to site, and [`OpSum`](@ref) checks every operator against its own
`ds[pos]`. Terms are combined into sums with `+`
(`OpTerm + OpTerm -> OpSum`, `OpSum + OpTerm`, `OpTerm + OpSum`, `OpSum + OpSum`),
multiplied by a number (`x * term`, `term * x`) and evaluated at a real time
(`term(t)` freezes the coefficient at `t`).
"""
struct OpTerm{T<:Number}
	coeff::Coeff
	positions::Vector{Int}
	operators::Vector{Matrix{T}}
	function OpTerm(coeff, positions::AbstractVector{Int},
					operators::AbstractVector{<:AbstractMatrix})
		N = length(positions)
		(N > 0) || throw(ArgumentError("term must act on at least one site"))
		(length(operators) == N) ||
			throw(ArgumentError("got $(length(operators)) operators for $N positions"))
		allunique(positions) ||
			throw(ArgumentError("a term cannot act on the same position twice"))
		for op in operators
			(size(op, 1) == size(op, 2)) ||
				throw(DimensionMismatch("every operator of a term must be a square matrix, got $(size(op))"))
		end
		coeff isa Coeff || (coeff = Coeff(coeff))
		# the storage type promotes the operators with the coefficient's scalar type
		# (a function coefficient is probed at t = 0)
		T = promote_type(scalartype(coeff), scalartype(operators[1]))
		for op in operators
			T = promote_type(T, scalartype(op))
		end
		isconstant(coeff) && (coeff = Coeff(convert(T, coeff.f)))
		perm = sortperm(positions)
		return new{T}(coeff, collect(Int, positions[perm]),
			[convert(Matrix{T}, op) for op in operators[perm]])
	end
end

scalartype(::Type{OpTerm{T}}) where {T} = T
# the dynamically probed scalar type of a term: the operators' type promoted with the
# coefficient's produced number type (a function coefficient is probed at t = 0)
scalartype(t::OpTerm) = promote_type(scalartype(t.operators[1]), scalartype(t.coeff))

"time-dependent coefficients are only supported by `trotter_propagator`"
function _check_constant(t::OpTerm)
	isconstant(t) || throw(ArgumentError(
		"time-dependent coefficients are only supported by `trotter_propagator`"))
	return t
end

Base.:(==)(a::OpTerm, b::OpTerm) = a.coeff == b.coeff && a.positions == b.positions &&
	a.operators == b.operators

# the hermitian conjugate: conjugated coefficient, adjointed operators (positions keep
# their order — each local operator is adjointed in place)
Base.adjoint(t::OpTerm) = OpTerm(_adjoint(t.coeff), t.positions, adjoint.(t.operators))

# scalar multiplication scales the coefficient
Base.:*(x::Number, t::OpTerm) = OpTerm(x * t.coeff, t.positions, t.operators)
Base.:*(t::OpTerm, x::Number) = x * t
Base.:-(t::OpTerm) = -1 * t

# freeze the coefficient at the real time `tval`
(t::OpTerm)(tval::Real) = OpTerm(t.coeff(tval), t.positions, t.operators)

OpTerm(coeff, pairs::Pair{Int,<:AbstractMatrix}...) =
	OpTerm(coeff, Int[first.(pairs)...], AbstractMatrix[last.(pairs)...])

"""
	term(coeff, i₁ => O₁, ...)
	term(i₁ => O₁, ...)

Convenience constructor of [`OpTerm`](@ref) (coefficient defaults to 1).
"""
term(coeff, pairs::Pair{Int,<:AbstractMatrix}...) = OpTerm(coeff, pairs...)
term(pairs::Pair{Int,<:AbstractMatrix}...) = term(one(Float64), pairs...)

"""
	OpSum(ds)
	OpSum(ds, terms)
	OpSum(ds, terms...)

A validated collection of [`OpTerm`](@ref)s on a lattice with physical dimensions
`ds::Vector{Int}`. The terms are validated against `ds` on construction (positions in
range, every operator matching the local dimension), so an `OpSum` is self-consistent
by construction; it is the input of `MPOHamiltonian(::OpSum)`. The scalar type `T` is
the promoted scalar type of the term operators and coefficients (a time-dependent
[`Coeff`](@ref) keeps its own function). An empty `OpSum(ds)` serves as the starting
point of an incremental build: terms are accumulated with `+=`
(`OpSum + OpTerm -> OpSum`), every added term is validated against the lattice. Terms
can also be summed lattice-free (`OpTerm + OpTerm -> OpSum`, `OpTerm + OpSum`,
`OpSum + OpSum`), in which case `ds` is inferred from the operators (sites without
operators default to the first term's dimension). A sum can be multiplied by a number
(`x * sum`, `sum * x`) and evaluated at a real time (`sum(t)` freezes every
coefficient at `t`).
"""
struct OpSum{T<:Number}
	data::Vector{OpTerm{T}}
	ds::Vector{Int}
end

OpSum(ds::AbstractVector{Int}) = OpSum{Float64}(OpTerm{Float64}[], collect(Int, ds))

scalartype(::Type{OpSum{T}}) where {T} = T
# dynamically probed: promote over the terms (function coefficients probed at t = 0)
scalartype(s::OpSum) = isempty(s.data) ? Float64 :
	reduce((a, b) -> promote_type(a, b), (scalartype(t) for t in s.data))

"""
	isconstant(term)
	isconstant(sum)

Whether the coefficient(s) of an [`OpTerm`](@ref) / [`OpSum`](@ref) are
time-independent. Time-dependent terms are only supported by
[`trotter_propagator`](@ref); every other consumer requires `isconstant == true`.
"""
isconstant(t::OpTerm) = isconstant(t.coeff)
isconstant(s::OpSum) = all(isconstant, s.data)

phydims(x::OpSum) = x.ds
phydim(x::OpSum, site) = x.ds[site]

Base.:(==)(a::OpSum, b::OpSum) = a.ds == b.ds && a.data == b.data

Base.adjoint(s::OpSum) = OpSum(s.ds, adjoint.(s.data))

# scalar multiplication scales every term
Base.:*(x::Number, s::OpSum) =
	isempty(s.data) ? OpSum(copy(s.ds)) : OpSum(s.ds, [x * t for t in s.data])
Base.:*(s::OpSum, x::Number) = x * s
Base.:-(s::OpSum) = -1 * s

# freeze every coefficient at the real time `tval`
(s::OpSum)(tval::Real) =
	isempty(s.data) ? OpSum(copy(s.ds)) : OpSum(s.ds, [t(tval) for t in s.data])

_convert_term(::Type{T}, t::OpTerm{T}) where {T} = t
_convert_term(::Type{T}, t::OpTerm) where {T} =
	OpTerm(_convert_coeff(T, t.coeff), t.positions,
		   [convert(Matrix{T}, op) for op in t.operators])
_convert_coeff(::Type{T}, c::Coeff) where {T} = isconstant(c) ? Coeff(convert(T, c.f)) : c

# validated assembly of an OpSum from a lattice and a term list (scalar types promoted)
function _opsum(ds::Vector{Int}, data::Vector{<:OpTerm})
	T = Float64
	for t in data
		T = promote_type(T, scalartype(t))
	end
	terms = [_convert_term(T, t) for t in data]
	for t in terms, (pos, op) in zip(t.positions, t.operators)
		(1 <= pos <= length(ds)) ||
			throw(ArgumentError("term position $pos out of range for $(length(ds)) sites"))
		(size(op, 1) == ds[pos]) ||
			throw(DimensionMismatch("operator dimension $(size(op, 1)) does not match ds[$pos] = $(ds[pos])"))
	end
	return OpSum{T}(terms, ds)
end

OpSum(ds::AbstractVector{Int}, terms::AbstractVector{<:OpTerm}) = _opsum(collect(Int, ds), terms)
OpSum(ds::AbstractVector{Int}, terms::OpTerm...) = _opsum(collect(Int, ds), collect(terms))

# the lattice of a sum built with `+`: the largest site position, with the local
# dimension of the first operator acting on each site; sites without operators default
# to the first term's dimension (the uniform-lattice assumption). An existing OpSum
# contributes its lattice; a wider term grows it.
function _opsum_dims(ts...)
	dims = Dict{Int, Int}()
	L = 0
	for t in ts
		t isa OpSum && (L = max(L, length(t.ds)))
	end
	for t in ts
		if t isa OpSum
			for (p, d) in enumerate(t.ds)
				dims[p] = get(dims, p, d)
				dims[p] == d || throw(DimensionMismatch(
					"conflicting local dimensions at site $p: $(dims[p]) != $d"))
			end
		else
			for (pos, op) in zip(t.positions, t.operators)
				L = max(L, pos)
				d = get(dims, pos, size(op, 1))
				d == size(op, 1) || throw(DimensionMismatch(
					"conflicting local dimensions at site $pos: $d != $(size(op, 1))"))
				dims[pos] = d
			end
		end
	end
	d0 = first(values(dims))
	ds = [get(dims, p, d0) for p in 1:L]
	return ds
end

Base.:+(t1::OpTerm, t2::OpTerm) = _opsum(_opsum_dims(t1, t2), [t1, t2])
Base.:+(t::OpTerm, s::OpSum) = _opsum(_opsum_dims(t, s), [t, s.data...])
Base.:+(s::OpSum, t::OpTerm) = _opsum(_opsum_dims(t, s), [s.data..., t])
Base.:+(s1::OpSum, s2::OpSum) = _opsum(_opsum_dims(s1, s2), [s1.data..., s2.data...])
