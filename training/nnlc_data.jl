module NNLCData

using DataFrames
using Statistics

export NNLC_INPUT_COLUMNS, prepare_nnlc_training_data, validate_model_direction,
       validate_torque_direction

const NNLC_INPUT_COLUMNS = [
  :v_ego,
  :actual_lateral_accel,
  :lateral_jerk,
  :roll,
  :actual_lateral_accel_tm03,
  :actual_lateral_accel_tm02,
  :actual_lateral_accel_tm01,
  :actual_lateral_accel_tp03,
  :actual_lateral_accel_tp06,
  :actual_lateral_accel_tp10,
  :actual_lateral_accel_tp15,
  :roll_tm03,
  :roll_tm02,
  :roll_tm01,
  :roll_tp03,
  :roll_tp06,
  :roll_tp10,
  :roll_tp15,
]

function validate_torque_direction(data::DataFrame)::Float64
  direction_correlation = cor(data.actual_lateral_accel, data.torque_output)
  if !isfinite(direction_correlation) || direction_correlation <= 0.0
    throw(ArgumentError(
      "torque_output has the wrong NNLC sign: correlation with actual_lateral_accel " *
      "must be positive, got $(round(direction_correlation, digits=6))",
    ))
  end
  return direction_correlation
end

function validate_model_direction(output_samples)::Nothing
  for (speed, negative_output, neutral_output, positive_output) in output_samples
    if !(negative_output < neutral_output < positive_output)
      throw(ArgumentError(
        "trained NNLC model has the wrong torque direction at $(speed) m/s: " *
        "outputs for lateral acceleration -1/0/+1 were " *
        "$(negative_output), $(neutral_output), $(positive_output)",
      ))
    end
  end
  return nothing
end

function prepare_nnlc_training_data(data::DataFrame)::DataFrame
  prepared = copy(data)

  # tp03 is 0.3 seconds in the future. This finite difference must use
  # 0.3 s, not the 0.03 s value used by the previous pipeline.
  prepared[!, :lateral_jerk] = @. (
    prepared.actual_lateral_accel_tp03 - prepared.actual_lateral_accel
  ) / 0.3

  select!(prepared, vcat(NNLC_INPUT_COLUMNS, [:torque_output]))
  validate_torque_direction(prepared)
  return prepared
end

end
