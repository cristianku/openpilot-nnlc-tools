using Test
using DataFrames

include(joinpath(@__DIR__, "..", "training", "nnlc_data.jl"))
using .NNLCData


function sample_data(; torque_output=Float64[-0.3, 0.3])
  lateral_accel = Float64[-1.0, 1.0]
  data = DataFrame(
    v_ego=Float64[20.0, 20.0],
    actual_lateral_accel=lateral_accel,
    roll=zeros(2),
    torque_output=torque_output,
  )

  for suffix in ("tm03", "tm02", "tm01", "tp03", "tp06", "tp10", "tp15")
    data[!, Symbol("actual_lateral_accel_" * suffix)] = copy(lateral_accel)
    data[!, Symbol("roll_" * suffix)] = zeros(2)
  end
  data.actual_lateral_accel_tp03 .= lateral_accel .+ 0.3
  return data
end


@testset "NNLC training data contract" begin
  prepared = prepare_nnlc_training_data(sample_data())

  @test names(prepared) == [
    "v_ego",
    "actual_lateral_accel",
    "lateral_jerk",
    "roll",
    "actual_lateral_accel_tm03",
    "actual_lateral_accel_tm02",
    "actual_lateral_accel_tm01",
    "actual_lateral_accel_tp03",
    "actual_lateral_accel_tp06",
    "actual_lateral_accel_tp10",
    "actual_lateral_accel_tp15",
    "roll_tm03",
    "roll_tm02",
    "roll_tm01",
    "roll_tp03",
    "roll_tp06",
    "roll_tp10",
    "roll_tp15",
    "torque_output",
  ]
  @test prepared.lateral_jerk ≈ Float64[1.0, 1.0]
end

@testset "NNLC torque direction validation" begin
  @test_throws ArgumentError prepare_nnlc_training_data(
    sample_data(torque_output=Float64[0.3, -0.3]),
  )
end

@testset "NNLC model direction validation" begin
  @test validate_model_direction([
    (10.0, -0.4, 0.0, 0.4),
    (20.0, -0.6, 0.0, 0.6),
  ]) === nothing

  @test_throws ArgumentError validate_model_direction([
    (20.0, 0.6, 0.0, -0.6),
  ])
end
