namespace Gpmd

def half : Rat := 1 / 2
def two : Rat := 2
def four : Rat := 4

theorem density_ratio_coefficient : (2 : Rat) / 4 = half := by
  native_decide

theorem log_ratio_coeff_on_one :
    - ((2 : Rat) * 1) / (4 * 1) = -half := by
  native_decide

theorem theta_star_pos : (0 : Rat) < half := by native_decide
theorem theta_star_lt_two : half < two := by native_decide
theorem theta_star_in_window : (0 : Rat) < half ∧ half < two :=
  ⟨theta_star_pos, theta_star_lt_two⟩

theorem dimensionless_example_d4_gap8 :
    (4 : Rat) * (half * 8 / 4) / 8 = half := by
  native_decide

theorem inv_theta_half_at_one :
    (1 : Rat) / 1 - half = (two - 1) / (two * 1) := by
  native_decide

theorem inv_theta_half_at_four :
    (1 : Rat) / 4 - half = (two - 4) / (two * 4) := by
  native_decide

theorem factor_pos_when_theta_half : (0 : Rat) < (two - half) / (two * half) := by
  native_decide

theorem factor_neg_when_theta_four : (two - 4) / (two * 4) < (0 : Rat) := by
  native_decide

theorem factor_zero_when_theta_two : (two - two) / (two * two) = (0 : Rat) := by
  native_decide

end Gpmd
