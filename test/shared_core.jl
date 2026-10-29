import MOSSQuantumMPS
const SharedMPSCore = MOSSQuantumMPS.MPSCore
const FMA = FiniteMPSAlgorithms

@testset "shared MPSCore contract" begin
    @test FMA.CanonicalMPS === SharedMPSCore.CanonicalMPS
    @test FMA.CanonicalMPO === SharedMPSCore.CanonicalMPO
    @test FMA.MPOHamiltonian === SharedMPSCore.MPOHamiltonian
    @test FMA.NoTruncation === SharedMPSCore.NoTruncation
    @test FMA.apply! === SharedMPSCore.apply!
    @test FMA.expectation === SharedMPSCore.expectation
    @test FMA.bonddims === SharedMPSCore.bonddims
    @test FMA.space_l === SharedMPSCore.space_l
    @test FMA.space_r === SharedMPSCore.space_r
    @test FMA.phydim === SharedMPSCore.phydim
    state = MOSSQuantumMPS.zero_state_mps(3)
    @test state.core isa FMA.CanonicalMPS
    @test FMA.bonddims(state.core) == MOSSQuantumMPS.bonddims(state)
end
