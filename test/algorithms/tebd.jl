# dense L-site matrix of the documented (i1',...,iN',i1,...,iN) gate tensor on sites (i, i+1)
function two_site_dense(op4, i, L, d::Int=2)
	M = reshape(permutedims(op4, (2, 1, 4, 3)), d^2, d^2)   # (i1 i2)',(i1 i2), i1 slowest
	I2 = Matrix{ComplexF64}(I, d, d)
	K = Matrix{ComplexF64}(I, 1, 1)
	k = 1
	while k <= L
		if k == i
			K = kron(K, M)
			k += 2
		else
			K = kron(K, I2)
			k += 1
		end
	end
	return K
end

# generalized gate_tensor: kron-convention matrix (uniform local dim d) → the
# documented N-site gate tensor (support site 1 the slowest index)
function gate_tensorN(M::AbstractMatrix, N::Int, d::Int=2)
	return permutedims(reshape(M, ntuple(_ -> d, Val(2N))...),
					   (ntuple(i -> N + 1 - i, Val(N))..., ntuple(i -> 2N + 1 - i, Val(N))...))
end

# dense L-site matrix of the documented gate tensor on the NON-adjacent pair (i, j)
# (identities on the legs in between; supports j = i + 3, the tested case)
function two_site_dense_pair(op4, i, j, L, d::Int=2)
	j - i == 3 || throw(ArgumentError("unsupported gap"))
	δ = Matrix{ComplexF64}(I, d, d)
	# block over sites i..j; the row/column index must be ordered FASTEST-first for the
	# column-major reshape (dims run p_j, m2, m1, p_i | q_j, n2, n1, q_i), so that site i
	# is the slowest index inside the block, matching the adjacent kron embedding
	@tensor B[pj, m2, m1, p1, qj, n2, n1, q1] := op4[p1, pj, q1, qj] * δ[m1, n1] * δ[m2, n2]
	K = reshape(B, d^4, d^4)
	I2 = Matrix{ComplexF64}(I, d, d)
	return kron(vcat(fill(I2, i - 1), [K], fill(I2, L - j))...)
end


@testset "tebd" begin
	Random.seed!(55)
	# GateOp stores the op verbatim and determines `isunitary` by default;
	# duplicate positions rejected
	h = randn(4, 4) + im * randn(4, 4)
	gg0 = GateOp((1, 2), gate_tensor(h))             # non-unitary input accepted
	@test gg0 isa AbstractGate
	@test !isunitary(gg0)
	@test_throws ArgumentError GateOp((2, 2), gate_tensor(h))
	# unsorted positions are sorted internally and the operator legs follow along
	gg1 = GateOp((2, 1), gate_tensor(h))
	@test positions(gg1) == (1, 2)
	@test gg1.op ≈ permutedims(gate_tensor(h), (2, 1, 4, 3))
	hh = h + h'                          # hermitian → exp(-im·hh) is unitary
	U = exp(Matrix(-im * hh))
	g = GateOp((1, 2), gate_tensor(U))
	@test positions(g) == (1, 2)
	# adjoint swaps the ket/bra index blocks and conjugates
	gt = adjoint(g)
	@test gt.op ≈ permutedims(conj(g.op), (3, 4, 1, 2))
	# isunitary predicate: checked automatically for the fused operator matrix …
	@test isunitary(g)
	@test !isunitary(GateOp((1, 2), gate_tensor(h)))
	r1 = randn(2, 2) + im * randn(2, 2)
	@test isunitary(GateOp((3,), exp(Matrix(-im * (r1 + r1')))))
	@test !isunitary(GateOp((3,), randn(ComplexF64, 2, 2)))
	# … trusted verbatim when the keyword is given …
	@test isunitary(GateOp((1, 2), gate_tensor(h); isunitary=true))
	@test !isunitary(GateOp((1, 2), gate_tensor(U); isunitary=false))
	@test isunitary(shift(GateOp((1, 2), gate_tensor(U)), 2))
	@test !isunitary(shift(GateOp((1, 2), gate_tensor(h)), 2))
	# … and controlled by `atol`
	I2 = Matrix{ComplexF64}(I, 2, 2)
	@test !isunitary(GateOp((1,), (1 + 1e-6) * I2; atol=1e-10))
	@test isunitary(GateOp((1,), (1 + 1e-6) * I2; atol=1e-4))

	# GateOp stores the given op verbatim (caller builds exp);
	# gate_tensor converts the Kronecker-convention matrix to the documented tensor
	gg = GateOp((2, 3), gate_tensor(exp(Matrix(-im * hh * 0.1))))
	@test gg.op ≈ permutedims(reshape(exp(Matrix(-im * hh * 0.1)), 2, 2, 2, 2), (2, 1, 4, 3)) atol = 1e-12

	# apply! of an exact unitary vs dense contraction
	L = 6
	ψ = randommps(ComplexF64, fill(2, L); D=8)
	canonicalize!(ψ)
	@test iscanonical(ψ)
	ψref = todense(ψ)
	Vd = two_site_dense(gg.op, 2, L)
	apply!(gg, ψ; trunc=NoTruncation())
	ψnew = todense(ψ)
	@test ψnew ≈ Vd * ψref atol = 1e-10
	# right-canonical form preserved without a trailing canonicalize!
	@test isrightcanonical(ψ[1])
	@test iscanonical(ψ; atol=1e-8)
	@test norm(ψ) ≈ 1 atol = 1e-10

	# a whole TEBD layer of gates keeps the state canonical (Hastings auto-canonical behavior)
	ψ2 = randommps(ComplexF64, fill(2, L); D=8)
	canonicalize!(ψ2)
	vref = todense(ψ2)
	gateAt = Dict{Int,Matrix{ComplexF64}}()
	for i in 1:2:L-1
		ggl = GateOp((i, i + 1), gate_tensor(exp(Matrix(-im * hh * 0.07))))
		gateAt[i] = reshape(permutedims(ggl.op, (2, 1, 4, 3)), 4, 4)
		apply!(ggl, ψ2; trunc=NoTruncation())
	end
	@test iscanonical(ψ2; atol=1e-8)
	vlayer = copy(vref)
	for i in 1:2:L-1
		K = Matrix{ComplexF64}(I, 1, 1)
		k = 1
		while k <= L
			if k == i
				K = kron(K, gateAt[i]); k += 2
			else
				K = kron(K, Matrix{ComplexF64}(I, 2, 2)); k += 1
			end
		end
		vlayer = K * vlayer
	end
	@test todense(ψ2) ≈ vlayer atol = 1e-10

	# swap! exchanges the physical content of the neighboring sites i, i+1 (the
	# Hastings SWAP gate): the dense amplitude (x1, x2, ...) moves to (x2, x1, ...)
	ψa = randommps(ComplexF64, fill(2, L); D=8)
	canonicalize!(ψa)
	ψa0 = todense(ψa)
	n0 = norm(ψa)
	swap!(ψa, 2; trunc=NoTruncation())
	# todense flattens with site 1 the SLOWEST index, so a column-major reshape has
	# dim k = site L+1-k; exchanging sites 2,3 means exchanging dims L-1,L-2
	vt = reshape(ψa0, fill(2, L)...)
	perm = collect(1:L)
	perm[L-1], perm[L-2] = perm[L-2], perm[L-1]
	vexp = vec(permutedims(vt, Tuple(perm)))
	@test norm(todense(ψa) - vexp) / norm(vexp) < 1e-10
	@test norm(ψa) ≈ n0 atol = 1e-10
	@test isrightcanonical(ψa[1])
	@test iscanonical(ψa; atol=1e-8)

	# sequences of adjacent swaps (permutation2swaps) realize the site permutation:
	# the inverse sequence brings back the original physical content exactly
	ψb = randommps(ComplexF64, fill(2, L); D=8)
	canonicalize!(ψb)
	ψb0 = copy(ψb)
	vb0 = todense(ψb)
	perm = randperm(L)
	for b in FiniteMPSAlgorithms.permutation2swaps(perm)
		swap!(ψb, b; trunc=NoTruncation())
	end
	@test iscanonical(ψb; atol=1e-8)
	@test todense(ψb) != vb0 || isperm(perm) && perm == collect(1:L)   # content moved
	for b in FiniteMPSAlgorithms.permutation2swaps(invperm(perm))
		swap!(ψb, b; trunc=NoTruncation())
	end
	@test iscanonical(ψb; atol=1e-8)
	@test norm(todense(ψb) - vb0) / norm(vb0) < 1e-10
	@test norm(ψb) ≈ norm(ψb0) atol = 1e-10
end

@testset "gate application (possibly non-unitary)" begin
	Random.seed!(77)
	L = 6
	# a non-unitary op is accepted and stored verbatim
	h = randn(4, 4) + im * randn(4, 4)
	gg = GateOp((1, 2), gate_tensor(h))
	@test gg.op ≈ permutedims(reshape(Matrix{ComplexF64}(h), 2, 2, 2, 2), (2, 1, 4, 3))
	# adjoint swaps the ket/bra index blocks and conjugates
	@test adjoint(gg).op ≈ permutedims(conj(gg.op), (3, 4, 1, 2))

	# apply! with canonicalize=false: the physical state matches the dense
	# application; the canonical form is not restored after a non-unitary gate
	ψ = randommps(ComplexF64, fill(2, L); D=8)
	canonicalize!(ψ)
	ψref = todense(ψ)
	apply!(gg, ψ; trunc=NoTruncation(), canonicalize=false)
	K = two_site_dense(gg.op, 1, L)
	@test todense(ψ) ≈ K * ψref atol = 1e-10
	@test norm(ψ) ≈ norm(K * ψref) atol = 1e-10

	# the default canonicalize = !isunitary(g) re-canonicalizes after this gate
	ψc = randommps(ComplexF64, fill(2, L); D=8)
	canonicalize!(ψc)
	ψcref = todense(ψc)
	apply!(gg, ψc; trunc=NoTruncation())
	@test todense(ψc) ≈ K * ψcref atol = 1e-10
	@test iscanonical(ψc; atol=1e-8)

	# a unitary gate behaves identically (and keeps the canonical form on its own)
	ψu = randommps(ComplexF64, fill(2, L); D=8)
	canonicalize!(ψu)
	ψuref = todense(ψu)
	hh = h + h'
	gu = GateOp((3, 4), gate_tensor(exp(Matrix(-im * hh))))
	@test isunitary(gu)
	Vd = two_site_dense(gu.op, 3, L)
	apply!(gu, ψu; trunc=NoTruncation())
	@test todense(ψu) ≈ Vd * ψuref atol = 1e-10
	@test iscanonical(ψu; atol=1e-8)
	@test norm(ψu) ≈ 1 atol = 1e-10
end

@testset "gates coverage" begin
    Random.seed!(88)
    hh = randn(4, 4) + im * randn(4, 4)
    hh = hh + hh'
    U = exp(Matrix(-im * hh))
    g = GateOp((1, 2), gate_tensor(U))
    @test g isa AbstractGate
    @test GateOp((1, 2), gate_tensor(exp(Matrix(-im * hh * 0.1)))) isa AbstractGate
    # shift: move gate by 2 sites
    g2 = shift(g, 2)
    @test positions(g2) == (3, 4)
    @test g2.op ≈ g.op
    # shift back
    g3 = shift(g2, -2)
    @test positions(g3) == (1, 2)
    # shift a possibly non-unitary gate
    gg = GateOp((1, 2), gate_tensor(exp(Matrix(-im * hh * 0.1))))
    gg2 = shift(gg, 1)
    @test positions(gg2) == (2, 3)
end

@testset "multi-site gates" begin
	Random.seed!(66)
	L = 6
	d = 2
	ψ = randommps(ComplexF64, fill(d, L); D=8)
	canonicalize!(ψ)
	v0 = todense(ψ)

	# dense L-site matrix of an N-site gate tensor on the ascending `positions`
	# (identities on the complementary sites; plain index loops, todense convention)
	function gate_dense(op, positions)
		N = length(positions)
		dn = d^N
		# kron-convention matrix (support site 1 most significant): rows = outputs
		M = reshape(permutedims(op, (ntuple(i -> N + 1 - i, Val(N))...,
				ntuple(i -> 2N + 1 - i, Val(N))...)), dn, dn)
		K = zeros(ComplexF64, d^L, d^L)
		for col in 0:(d^L-1), row in 0:(d^L-1)
			rbits = [(row >> (L - t)) & 1 for t in 1:L]
			cbits = [(col >> (L - t)) & 1 for t in 1:L]
			all(t -> t in positions || rbits[t] == cbits[t], 1:L) || continue
			mrow = sum(rbits[positions[m]] << (N - m) for m in 1:N)
			mcol = sum(cbits[positions[m]] << (N - m) for m in 1:N)
			K[row+1, col+1] = M[mrow+1, mcol+1]
		end
		return K
	end

	# contiguous 3-site gate on (2, 3, 4)
	hh3 = randn(8, 8) + im * randn(8, 8)
	U3 = exp(Matrix(-im * (hh3 + hh3')))
	g3 = GateOp((2, 3, 4), gate_tensorN(U3, 3))
	ψ3 = copy(ψ)
	apply!(g3, ψ3; trunc=NoTruncation())
	@test todense(ψ3) ≈ gate_dense(g3.op, (2, 3, 4)) * v0 atol = 1e-10
	@test iscanonical(ψ3; atol=1e-8)                    # unitary: gauge preserved
	@test norm(ψ3) ≈ 1 atol = 1e-10

	# non-adjacent 4-site gate on (1, 3, 4, 6): gather, apply on the block, scatter
	hh4 = randn(16, 16) + im * randn(16, 16)
	U4 = exp(Matrix(-im * (hh4 + hh4')))
	g4 = GateOp((1, 3, 4, 6), gate_tensorN(U4, 4))
	ψ4 = copy(ψ)
	apply!(g4, ψ4; trunc=NoTruncation())
	@test todense(ψ4) ≈ gate_dense(g4.op, (1, 3, 4, 6)) * v0 atol = 1e-9
	@test iscanonical(ψ4; atol=1e-8)
	@test norm(ψ4) ≈ 1 atol = 1e-10

	# unsorted positions are sorted internally (legs follow): the input legs here are
	# given in the (6, 4, 3, 1) order, so the sorted gate equals g4
	g4s = GateOp((6, 4, 3, 1), permutedims(g4.op, (4, 3, 2, 1, 8, 7, 6, 5)))
	@test positions(g4s) == (1, 3, 4, 6)
	@test g4s.op ≈ g4.op
	ψ4s = copy(ψ)
	apply!(g4s, ψ4s; trunc=NoTruncation())
	@test todense(ψ4s) ≈ todense(ψ4) atol = 1e-10

	# non-unitary 3-site gate: exact state, canonicity restored by the default
	# canonicalize = !isunitary(g)
	gn = GateOp((2, 4, 5), gate_tensorN(randn(ComplexF64, 8, 8), 3))
	@test !isunitary(gn)
	ψn = copy(ψ)
	apply!(gn, ψn; trunc=NoTruncation())
	@test todense(ψn) ≈ gate_dense(gn.op, (2, 4, 5)) * v0 atol = 1e-9
	@test iscanonical(ψn; atol=1e-8)
	ψc = copy(ψ)
	apply!(gn, ψc; trunc=NoTruncation(), canonicalize=false)
	@test todense(ψc) ≈ todense(ψn) atol = 1e-9

	# nonuniform physical dims: full-support 3-site gate on (1, 2, 3), ds = (2, 3, 4)
	dsn = [2, 3, 4]
	ψu = randommps(ComplexF64, dsn; D=4)
	canonicalize!(ψu)
	vu = todense(ψu)
	hu = randn(ComplexF64, 24, 24)
	# documented legs (o1, o2, o3 | i1, i2, i3) from the kron-convention matrix
	gu = GateOp((1, 2, 3), permutedims(reshape(hu, 4, 3, 2, 4, 3, 2), (3, 2, 1, 6, 5, 4)))
	apply!(gu, ψu; trunc=NoTruncation())
	# todense order (site 1 slowest) from the op legs (o1, o2, o3 | i1, i2, i3)
	Mu = reshape(permutedims(gu.op, (3, 2, 1, 6, 5, 4)), 24, 24)
	@test todense(ψu) ≈ Mu * vu atol = 1e-10
end

@testset "fuse_gates" begin
	Random.seed!(67)
	L = 6
	hh = randn(4, 4) + im * randn(4, 4)
	hh += hh'

	# three overlapping gates merge into one on the union support (2, 3)
	r2 = randn(2, 2) + im * randn(2, 2); r2 += r2'
	r3 = randn(2, 2) + im * randn(2, 2); r3 += r3'
	g1 = GateOp((2,), exp(Matrix(-im * 0.2 * r2)))
	g2 = GateOp((3,), exp(Matrix(-im * 0.3 * r3)))
	g3 = GateOp((2, 3), gate_tensor(exp(Matrix(-im * hh))))
	f = fuse_gates([g1, g2, g3])
	@test length(f) == 1
	@test positions(f[1]) == (2, 3)

	# the fused circuit applies the same operator product as the original
	ψ = randommps(ComplexF64, fill(2, L); D=4)
	canonicalize!(ψ)
	ψf = copy(ψ)
	ψg = copy(ψ)
	apply!(f, ψf; trunc=NoTruncation())
	apply!([g1, g2, g3], ψg; trunc=NoTruncation())
	@test todense(ψf) ≈ todense(ψg) atol = 1e-10
	@test iscanonical(ψf; atol=1e-8)

	# gates with disjoint supports commute and are not merged
	h1 = GateOp((1, 2), gate_tensor(exp(Matrix(-im * hh))))
	h2 = GateOp((4, 5), gate_tensor(exp(Matrix(-im * hh))))
	fd = fuse_gates([h1, h2])
	@test length(fd) == 2
	@test positions.(fd) == [(1, 2), (4, 5)]

	# the boundary gates of repeated Strang layers merge (the adjoint pass)
	hhA = randn(4, 4) + im * randn(4, 4); hhA += hhA'
	hhB = randn(4, 4) + im * randn(4, 4); hhB += hhB'
	half = [GateOp((1, 2), gate_tensor(exp(Matrix(-im * 0.1 * hhA)))),
			GateOp((2, 3), gate_tensor(exp(Matrix(-im * 0.1 * hhB))))]
	circ = vcat(half, reverse(half), half, reverse(half))
	fc = fuse_gates(circ)
	@test length(fc) < length(circ)
	ψa = copy(ψ)
	ψb = copy(ψ)
	apply!(fc, ψa; trunc=NoTruncation())
	apply!(circ, ψb; trunc=NoTruncation())
	@test todense(ψa) ≈ todense(ψb) atol = 1e-10

	# single-site gates are all absorbed once a multi-site gate covers every site
	single_gate(i) = GateOp((i,), exp(Matrix(-im * 0.1 * (a = randn(ComplexF64, 2, 2); a + a'))))
	singles = [single_gate(i) for i in 1:L]
	Hfull = randn(ComplexF64, 2^L, 2^L)
	Hfull += Hfull'
	gfull = GateOp(ntuple(identity, L), gate_tensorN(exp(Matrix(-im * 0.05 * Hfull)), L))
	@test isunitary(gfull)
	# the full-support gate sits in the middle of the single-site gates
	circ_all = vcat(singles[1:3], [gfull], singles[4:6])
	fa = fuse_gates(circ_all)
	@test length(fa) == 1
	@test positions(fa[1]) == ntuple(identity, L)
	@test isunitary(fa[1])
	ψa1 = copy(ψ)
	ψa2 = copy(ψ)
	apply!(fa, ψa1; trunc=NoTruncation())
	apply!(circ_all, ψa2; trunc=NoTruncation())
	@test todense(ψa1) ≈ todense(ψa2) atol = 1e-9
	@test iscanonical(ψa1; atol=1e-8)

	# a random mixed circuit (single-, two- and three-site gates, adjacent and not):
	# the fused circuit is exactly equivalent on a state and loses no gate
	rand_gate(pos::Tuple) = begin
		n = length(pos)
		a = randn(ComplexF64, 2^n, 2^n)
		GateOp(pos, gate_tensorN(exp(Matrix(-im * 0.1 / n * (a + a'))), n))
	end
	circ = AbstractGate[]
	for n in (1, 2, 3, 2, 1, 3, 2, 1, 2, 3)
		push!(circ, rand_gate(Tuple(sort!(randperm(L)[1:n]))))
	end
	@test all(isunitary, circ)
	fcirc = fuse_gates(circ)
	@test length(fcirc) <= length(circ)
	@test all(isunitary, fcirc)
	ψc1 = copy(ψ)
	ψc2 = copy(ψ)
	apply!(fcirc, ψc1; trunc=NoTruncation())
	apply!(circ, ψc2; trunc=NoTruncation())
	@test todense(ψc1) ≈ todense(ψc2) atol = 1e-9
	@test norm(ψc1) ≈ norm(ψc2) atol = 1e-10
end

@testset "TEBD vs exact diagonalization" begin
    Random.seed!(40)
    L = 4
    ds = fill(2, L)
    dim = 2^L
    # random hermitian nearest-neighbor couplings
    pairs = [(1, 2), (2, 3), (3, 4)]
    hs = [(randn(4, 4) + im * randn(4, 4)) for _ in pairs]
    hs = [(h + h') / 2 for h in hs]
    I2 = Matrix{ComplexF64}(LinearAlgebra.I, 2, 2)

    # dense H = Σ_i h_(i,i+1); the kron ordering keeps site 1 the slowest index,
    # matching todense. A nearest-neighbor h is ONE kron slot spanning sites (i, i+1).
    Hd = zeros(ComplexF64, dim, dim)
    for (k, (i, j)) in enumerate(pairs)
        factors = ntuple(a -> a == i ? hs[k] : I2, L - 1)
        Hd .+= reshape(kron(factors...), dim, dim)
    end
    ψ0 = randommps(ComplexF64, ds; D=8)
    v0 = todense(ψ0)

    # a single brickwall layer on disjoint sites (1,2) and (3,4): the two gates act on
    # independent factors and commute, so one Trotter layer is EXACT
    dt = 0.37
    ψ = copy(ψ0)
    apply!(GateOp((1, 2), gate_tensor(exp(-im * dt * hs[1]))), ψ; trunc=truncdimcutoff(64, 1e-12))
    apply!(GateOp((3, 4), gate_tensor(exp(-im * dt * hs[3]))), ψ; trunc=truncdimcutoff(64, 1e-12))
    Ulayer = kron(exp(-im * dt * hs[1]), exp(-im * dt * hs[3]))
    @test norm(todense(ψ) - Ulayer * v0) / norm(v0) < 1e-12

    # second-order Trotter time evolution vs exp(-i t H) from exact diagonalization
    dt = 0.01
    nsteps = 20
    G(k, τ) = GateOp((pairs[k][1], pairs[k][2]), gate_tensor(exp(-im * τ * hs[k])))
    ψt = copy(ψ0)
    for _ in 1:nsteps
        for k in 1:3
            apply!(G(k, dt / 2), ψt; trunc=truncdimcutoff(64, 1e-12))
        end
        for k in 3:-1:1
            apply!(G(k, dt / 2), ψt; trunc=truncdimcutoff(64, 1e-12))
        end
    end
    ψexact = exp(-im * nsteps * dt * Hd) * v0
    @test norm(todense(ψt) - ψexact) / norm(ψexact) < 1e-3
end

@testset "long-range gates and swaps" begin
    Random.seed!(77)
    L = 6
    ψ = randommps(ComplexF64, fill(2, L); D=4)
    canonicalize!(ψ)
    v0 = todense(ψ)

    # exchange the contents of two (generally non-adjacent) sites as a sequence of
    # adjacent swaps (equivalently: permutation2swaps of the transposition (i j))
    function exchange!(x, i, j; trunc=NoTruncation())
        lo, hi = minmax(i, j)
        for b in hi-1:-1:lo
            swap!(x, b; trunc)
        end
        for b in lo+1:hi-1
            swap!(x, b; trunc)
        end
        return x
    end

    # exchange!(ψ, i, j) moves the site contents: compare against the full-space
    # permutation that swaps the corresponding bits of every basis index
    i, j = 2, 5
    P = zeros(ComplexF64, 2^L, 2^L)
    for idx in 0:2^L-1
        bits = [(idx >> (L - t)) & 1 for t in 1:L]
        bits[i], bits[j] = bits[j], bits[i]
        jdx = sum(bits[t] << (L - t) for t in 1:L)
        P[jdx+1, idx+1] = 1.0
    end
    ψs = copy(ψ)
    exchange!(ψs, i, j)
    @test iscanonical(ψs; atol=1e-8)
    @test norm(todense(ψs) - P * v0) / norm(v0) < 1e-10
    # swapping back restores the original state exactly
    exchange!(ψs, i, j)
    @test norm(todense(ψs) - v0) / norm(v0) < 1e-10

    # adjacent-pair content swap
    ψadj = copy(ψ)
    exchange!(ψadj, 3, 4)
    Padj = zeros(ComplexF64, 2^L, 2^L)
    for idx in 0:2^L-1
        bits = [(idx >> (L - t)) & 1 for t in 1:L]
        bits[3], bits[4] = bits[4], bits[3]
        jdx = sum(bits[t] << (L - t) for t in 1:L)
        Padj[jdx+1, idx+1] = 1.0
    end
    @test norm(todense(ψadj) - Padj * v0) / norm(v0) < 1e-10

    # swapping a site with itself is a no-op
    ψid = copy(ψ)
    exchange!(ψid, 3, 3)
    @test todense(ψid) ≈ v0 atol = 1e-12

    # sites with different physical dimensions: the dimension list follows the content
    dsn = [2, 3, 2, 2]
    ψn = randommps(ComplexF64, dsn; D=3)
    canonicalize!(ψn)
    vn = todense(ψn)
    exchange!(ψn, 2, 4)
    @test phydims(ψn) == [2, 2, 2, 3]
    # column-major reshape: dim k = site L+1-k (fastest first); swapping sites 2,4
    # exchanges dims 3 and 1
    vt = reshape(vn, reverse(dsn)...)
    vref = vec(permutedims(vt, (3, 2, 1, 4)))
    @test norm(todense(ψn) - vref) / norm(vref) < 1e-10

    # long-range unitary gate on a non-adjacent pair vs the dense embedding
    hh = randn(4, 4) + im * randn(4, 4)
    U = exp(Matrix(-im * (hh + hh')))
    gu = GateOp((2, 5), gate_tensor(U))
    ψu = copy(ψ)
    apply!(gu, ψu; trunc=NoTruncation())
    Vu = two_site_dense_pair(gu.op, 2, 5, L)
    @test norm(todense(ψu) - Vu * v0) / norm(v0) < 1e-10
    # the content swaps are exact unitary re-gaugings: canonical form and norm preserved
    @test iscanonical(ψu; atol=1e-8)
    @test norm(ψu) ≈ 1 atol = 1e-10

    # long-range non-unitary gate vs the dense embedding (re-canonicalized)
    ggn = GateOp((2, 5), gate_tensor(randn(ComplexF64, 4, 4)))
    ψg = copy(ψ)
    apply!(ggn, ψg; trunc=NoTruncation(), canonicalize=true)
    Vg = two_site_dense_pair(ggn.op, 2, 5, L)
    @test norm(todense(ψg) - Vg * v0) / norm(Vg * v0) < 1e-10
    @test iscanonical(ψg; atol=1e-8)

    # truncation keeps the bond dimension bounded
    ψt = changebond!(copy(ψ); D=8)
    apply!(gu, ψt; trunc=truncdim(4))
    @test bonddim(ψt) <= 4
end

@testset "trotter_propagator" begin
	Random.seed!(68)
	L = 4
	d = 2
	I2 = Matrix{ComplexF64}(I, d, d)
	rh() = (A = randn(2, 2) + im * randn(2, 2); (A + A') / 2)

	# random hermitian nearest-neighbor hamiltonian from local product terms
	pairs = [(1, 2), (2, 3), (3, 4)]
	terms = OpTerm[]
	hb = Dict{Int,Matrix{ComplexF64}}()     # dense bond matrices (kron convention)
	for (k, (i, j)) in enumerate(pairs)
		A, B, C, D = rh(), rh(), rh(), rh()
		push!(terms, OpTerm(0.7, [i, j], [A, B]))
		push!(terms, OpTerm(-0.3, [i, j], [C, D]))
		hb[k] = 0.7 .* kron(A, B) .- 0.3 .* kron(C, D)
	end
	ham = OpSum(fill(d, L), terms)
	# dense H (kron convention, site 1 the slowest index, matching todense)
	Hd = zeros(ComplexF64, d^L, d^L)
	for (k, (i, j)) in enumerate(pairs)
		factors = ntuple(a -> a == i ? hb[k] : I2, L - 1)
		Hd .+= reshape(kron(factors...), d^L, d^L)
	end

	# one Strang step on a nearest-neighbor hamiltonian: even(τ/2), odd(τ), even(τ/2)
	t = 0.1
	gs = trotter_propagator(ham, (0, -im * t); order=2, stepsize=t)
	@test length(gs) == 4
	@test positions.(gs) == [(2, 3), (1, 2), (3, 4), (2, 3)]
	@test gs[1].op ≈ gate_tensor(exp(-im * (t / 2) * hb[2])) atol = 1e-12
	@test gs[2].op ≈ gate_tensor(exp(-im * t * hb[1])) atol = 1e-12

	# the circuit realizes exp(-im·t·H) on a random state (MPS vs ED)
	ψ0 = randommps(ComplexF64, fill(d, L); D=8)
	canonicalize!(ψ0)
	v0 = todense(ψ0)
	ψ = copy(ψ0)
	apply!(gs, ψ; trunc=NoTruncation())
	@test norm(todense(ψ) - exp(-im * t * Hd) * v0) / norm(v0) < 5e-3

	# multiple steps: accumulate layers of length stepsize
	t = 0.3
	gsm = trotter_propagator(ham, (0, -im * t); order=2, stepsize=0.1)
	@test length(gsm) == 12          # 3 steps × 4 gates
	ψm = copy(ψ0)
	apply!(gsm, ψm; trunc=NoTruncation())
	@test norm(todense(ψm) - exp(-im * t * Hd) * v0) / norm(v0) < 1e-2

	# 4th order (Suzuki fractal) is more accurate than 2nd order at equal time
	g2 = trotter_propagator(ham, (0, -im * t); order=2, stepsize=t)
	g4 = trotter_propagator(ham, (0, -im * t); order=4, stepsize=t)
	ψ2 = copy(ψ0)
	ψ4 = copy(ψ0)
	apply!(g2, ψ2; trunc=NoTruncation())
	apply!(g4, ψ4; trunc=NoTruncation())
	vex = exp(-im * t * Hd) * v0
	e2 = norm(todense(ψ2) - vex) / norm(vex)
	e4 = norm(todense(ψ4) - vex) / norm(vex)
	@test e4 < e2
	@test e4 < 1e-2

	# generic path: a three-site product term produces one gate per support at τ/2 + reversed
	A1, A2, A3 = rh(), rh(), rh()
	ham3 = ham + OpTerm(1.0, [1, 2, 3], [A1, A2, A3])
	gs3 = trotter_propagator(ham3, (0, -im * 0.1); order=2, stepsize=0.1)
	@test positions.(gs3) == [(1, 2), (1, 2, 3), (2, 3), (3, 4), (3, 4), (2, 3), (1, 2, 3), (1, 2)]
	h13 = kron(A1, kron(A2, A3))
	Hd3 = Hd + reshape(kron(h13, I2), d^L, d^L)
	ψ3 = copy(ψ0)
	apply!(gs3, ψ3; trunc=NoTruncation())
	@test norm(todense(ψ3) - exp(-im * 0.1 * Hd3) * v0) / norm(v0) < 5e-3

	# one-site terms are absorbed into the neighboring bond gates
	hz = randn(2, 2) + im * randn(2, 2)
	hz = (hz + hz') / 2
	hams = ham + OpTerm(1.0, [2], [hz])
	gss = trotter_propagator(hams, (0, -im * 0.1); order=2, stepsize=0.1)
	@test all(p -> length(p) == 2, positions.(gss))
	# the even bond (2, 3) carries the absorbed field: h_(2,3) + hz ⊗ I
	@test gss[1].op ≈ gate_tensor(exp(-im * 0.05 * (hb[2] + kron(hz, I2)))) atol = 1e-12
	Hds = Hd + reshape(kron(I2, kron(hz, I2), I2), d^L, d^L)
	ψs = copy(ψ0)
	apply!(gss, ψs; trunc=NoTruncation())
	@test norm(todense(ψs) - exp(-im * 0.1 * Hds) * v0) / norm(v0) < 5e-3

	# Coeff basics
	c1 = Coeff(2.5)
	c2 = Coeff(t -> 3.0 * t)
	@test isconstant(c1) && !isconstant(c2)
	@test c1(0.3) == 2.5 && c2(0.5) == 1.5
	@test scalartype(c1) == Float64 && scalartype(c2) == Float64
	@test Coeff(3) == 3 && 3 == Coeff(3) && Coeff(3.0) == 3.0

	# 3- and 4-site interactions: the 4th-order Suzuki propagator is far more accurate
	# than the 2nd-order one at equal time (NoTruncation → no bond-dimension error)
	L5 = 5
	ops3 = [rh(), rh(), rh()]
	ops4 = [rh(), rh(), rh(), rh()]
	ham5 = OpSum(fill(d, L5), [OpTerm(0.9, [1, 2, 3], ops3),
							   OpTerm(-0.6, [2, 3, 4, 5], ops4)])
	h3 = 0.9 .* kron(ops3...)                # sites (1, 2, 3), site 1 the most significant
	h4 = -0.6 .* kron(ops4...)               # sites (2, 3, 4, 5)
	Hd5 = kron(h3, I2, I2) + kron(I2, h4)
	t5 = 0.2
	g52 = trotter_propagator(ham5, (0, -im * t5); order=2, stepsize=t5)
	g54 = trotter_propagator(ham5, (0, -im * t5); order=4, stepsize=t5)
	ψ52 = randommps(ComplexF64, fill(d, L5); D=8)
	canonicalize!(ψ52)
	v05 = todense(ψ52)
	ψ54 = copy(ψ52)
	apply!(g52, ψ52; trunc=NoTruncation())
	apply!(g54, ψ54; trunc=NoTruncation())
	vex5 = exp(-im * t5 * Hd5) * v05
	e52 = norm(todense(ψ52) - vex5) / norm(vex5)
	e54 = norm(todense(ψ54) - vex5) / norm(vex5)
	@test e52 < 5e-2
	@test e54 < e52 / 10

	# time-dependent OpSum: the coefficients are frozen at the layer midpoints; for a
	# single (self-commuting) term the propagator is exp(∫f · H₀) up to the midpoint rule
	X = Float64[0 1; 1 0]
	Y = ComplexF64[0 -im; im 0]
	f(t) = 2.0 * exp(-1.5 * t)
	hamt = OpSum(fill(d, L), [OpTerm(f, [1, 2], [X, Y])])
	@test !isconstant(hamt) && isconstant(ham)
	tt = 0.4
	gt = trotter_propagator(hamt, (0.0, tt); order=2, stepsize=0.1)
	@test length(gt) == 4                    # 4 steps × 1 gate
	H0 = kron(kron(X, Y), I2, I2)
	@test gt[1].op ≈ gate_tensor(exp(0.1 * f(0.05) * kron(X, Y))) atol = 1e-12
	ψtd = copy(ψ0)
	apply!(gt, ψtd; trunc=NoTruncation())
	cint = 2.0 * (1 - exp(-1.5 * tt)) / 1.5  # ∫₀^tt f
	@test norm(todense(ψtd) - exp(cint * H0) * v0) / norm(v0) < 5e-3

	# every consumer except trotter_propagator rejects time-dependent terms
	@test_throws ArgumentError MPOHamiltonian(hamt)
	tdterm = OpTerm(f, [1], [X])
	ψg = randommps(ComplexF64, fill(d, L); D=4)
	canonicalize!(ψg)
	@test_throws ArgumentError expectationvalue(tdterm, ψg)
	@test_throws ArgumentError expectation(tdterm, ψg)

	# time-dependent hamiltonian: the accuracy improves with the Suzuki order as
	# expected — halving the step divides the error by ~4 (order 2) and ~16 (order 4).
	# Real-time evolution: the coefficients carry the -im factor, so H(t) is
	# anti-hermitian and the propagator is unitary
	hterms = OpTerm[]
	A1, B1, A2, B2, A3, A4, A5 = rh(), rh(), rh(), rh(), rh(), rh(), rh()
	push!(hterms, OpTerm(-1.0im, [1, 2], [A1, B1]))                            # constant
	push!(hterms, OpTerm(t -> -1.5im * exp(-0.7 * t), [2, 3], [A2, B2]))       # time-dependent
	push!(hterms, OpTerm(t -> -im * (0.8 + sin(t)), [1, 2, 3], [A3, A4, A5]))  # time-dependent
	hamtd = OpSum(fill(d, L), hterms)
	@test !isconstant(hamtd)
	ttot = 0.8
	ψ0b = randommps(ComplexF64, fill(d, L); D=8)
	canonicalize!(ψ0b)
	ref = copy(ψ0b)
	apply!(trotter_propagator(hamtd, (0, ttot); order=4, stepsize=ttot / 80), ref;
		   trunc=NoTruncation())
	vref = todense(ref)
	function tderr(order, dt)
		ψx = copy(ψ0b)
		apply!(trotter_propagator(hamtd, (0, ttot); order, stepsize=dt), ψx;
			   trunc=NoTruncation())
		return norm(todense(ψx) - vref) / norm(vref), norm(ψx)
	end
	e2c, n2c = tderr(2, 0.2)
	e2f, n2f = tderr(2, 0.1)
	e4c, n4c = tderr(4, 0.2)
	e4f, n4f = tderr(4, 0.1)
	@test e4c < e2c && e4f < e2f            # higher order is more accurate
	@test 2.5 < e2c / e2f < 6.5             # order 2: the error scales as dt²
	@test 9 < e4c / e4f < 26                # order 4: the error scales as dt⁴
	@test n2f ≈ 1 atol = 1e-10              # hermitian H(t) → unitary propagator

	# only even orders are supported
	@test_throws ArgumentError trotter_propagator(ham, (0, 1.0); order=3)
end

@testset "TEBD Lindblad (vectorized density matrix) vs ED" begin
    Random.seed!(43)
    L = 4
    d4 = fill(4, L)   # vectorized chain: one site carries (po, pi), po the slower index

    # model: H = -Σ h_i X_i - J Σ Z_i Z_{i+1}; hermitian jumps √γx·X on site 1, √γz·Z on site L
    h = [0.7, -0.4, 0.9, 0.2]; J = 0.6; γx = 0.35; γz = 0.55
    X = Float64[0 1; 1 0]; Z = Float64[1 0; 0 -1]
    I2 = Matrix{ComplexF64}(I, 2, 2); I4 = Matrix{ComplexF64}(I, 4, 4)

    # --- local superoperator generators in the MPS (interleaved) index basis ---
    # per-site index s = 2(po-1) + (pi-1): po is the slower quantum number; site 1 slowest
    field_gen(i) = -h[i] .* (kron(X, I2) - kron(I2, transpose(X)))   # -i[h_i X_i, ·]

    function zz_gen(i)                                               # -i[J Z_i Z_{i+1}, ·]
        tups = collect(Iterators.product(0:1, 0:1, 0:1, 0:1))        # (po', pi', po', pi')
        sidx(t) = (2 * t[1] + t[2]) * 4 + (2 * t[3] + t[4])
        G = zeros(ComplexF64, 16, 16)
        for rp in 1:16, cp in 1:16
            a, b, a2, b2 = tups[rp]
            c, d, c2, d2 = tups[cp]
            G[sidx((a, b, a2, b2)) + 1, sidx((c, d, c2, d2)) + 1] =
                -J * (Z[a+1, c+1] * Z[a2+1, c2+1] * (b == d) * (b2 == d2)
                      - Z[b+1, d+1] * Z[b2+1, d2+1] * (a == c) * (a2 == c2))
        end
        return G
    end

    # embed a two-site 16×16 block at bond (i, i+1) into the 4^L-dimensional chain
    function embed2(G, i)
        factors = Matrix{ComplexF64}[I4 for _ in 1:i-1]
        push!(factors, G)
        append!(factors, Matrix{ComplexF64}[I4 for _ in 1:L-i-1])
        return kron(factors...)
    end

    # hermitian single-site jump √γ·L: 𝒟(ρ) = γ(LρL - ½{L²,ρ}) = γ(kron(L,L) - I4) locally
    jump_gen(Lm, γ) = γ .* (kron(Lm, Lm) .- I4)

    # ---- ED reference: full Lindblad superoperator acting on the MPS-ordered vector ----
    super = zeros(ComplexF64, 4^L, 4^L)
    for i in 1:L
        g = field_gen(i)
        super .-= im .* kron([k == i ? g : I4 for k in 1:L]...)
    end
    for i in 1:L-1
        super .-= im .* embed2(zz_gen(i), i)
    end
    for (Lm, γ, s) in [(X, γx, 1), (Z, γz, L)]
        g = jump_gen(Lm, γ)
        super .+= kron([k == s ? g : I4 for k in 1:L]...)
    end

    # ---- MPS side: exact local exponential gates, Strang splitting ----
    # single-site generators embedded into bond gates (site L via the last bond);
    # gates on different sites commute, so the dissipator factor is exact
    fraw = [1 => kron(field_gen(1), I4), 2 => kron(field_gen(2), I4),
            3 => kron(field_gen(3), I4), 3 => kron(I4, field_gen(4))]
    even = [1 => zz_gen(1), 3 => zz_gen(3)]   # bonds (1,2), (3,4)
    odd = [2 => zz_gen(2)]                    # bond (2,3)
    diss = [1 => kron(jump_gen(X, γx), I4), 3 => kron(I4, jump_gen(Z, γz))]
    tr = truncdimcutoff(64, 1e-13)

    # U(τ) = e^{-i τ ad_H} via the symmetric local-gate sequence
    #   fields(τ/2), even(τ/2), odd(τ/2), odd(τ/2), even(τ/2), fields(τ/2)
    function apply_unitary!(ψ, τ)
        ev(pos, M) = apply!(GateOp((pos, pos + 1), gate_tensor(exp(Matrix(-im * (τ / 2) * M)))), ψ; trunc=tr)
        for (pos, M) in fraw; ev(pos, M); end
        for (pos, M) in even; ev(pos, M); end
        for (pos, M) in odd; ev(pos, M); end
        for (pos, M) in odd; ev(pos, M); end
        for (pos, M) in even; ev(pos, M); end
        for (pos, M) in fraw; ev(pos, M); end
    end

    # Strang step: U(dt/2) · e^{𝒟 dt} · U(dt/2); the dissipator is non-unitary, so the
    # chain is re-canonicalized after it
    function lindblad_step!(ψ, dt)
        apply_unitary!(ψ, dt / 2)
        for (pos, M) in diss
            apply!(GateOp((pos, pos + 1), gate_tensor(exp(Matrix(dt * M)))), ψ; trunc=tr, canonicalize=true)
        end
        apply_unitary!(ψ, dt / 2)
    end

    ψ = randommps(ComplexF64, d4; D=8)
    v0 = todense(ψ)
    dt, nsteps = 0.02, 12
    for _ in 1:nsteps
        lindblad_step!(ψ, dt)
    end

    v_exact = exp(super * (dt * nsteps)) * v0
    v_mps = todense(ψ)
    @test norm(v_mps - v_exact) / norm(v_exact) < 5e-3
end
