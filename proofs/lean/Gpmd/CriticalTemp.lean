import Gpmd.DensityRatio

namespace Gpmd

theorem operating_point_positive_factor : (0 : Rat) < (two - half) / (two * half) :=
  factor_pos_when_theta_half

theorem above_critical_negative_factor : (two - 4) / (two * 4) < (0 : Rat) :=
  factor_neg_when_theta_four

theorem at_critical_zero_factor : (two - two) / (two * two) = (0 : Rat) :=
  factor_zero_when_theta_two

end Gpmd
