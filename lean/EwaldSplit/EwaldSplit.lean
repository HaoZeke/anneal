import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.MeasureTheory.Measure.Lebesgue.Integral

open Real MeasureTheory Set Filter intervalIntegral

/-!
The Ewald split writes `1 = erf + erfc`. `erf` is the integral of the
Gaussian on a finite interval and `erfc` is the tail. The tail is the
real-space damping factor: it is `1` at the origin and tends to `0`
at infinity.
-/

theorem integral_exp_neg_sq : ∫ x : ℝ, exp (-x ^ 2) = √π := by
  simpa using integral_gaussian (1 : ℝ)

theorem integrable_exp_neg_sq : Integrable fun x : ℝ => exp (-x ^ 2) := by
  simpa using integrable_exp_neg_mul_sq (b := (1 : ℝ)) one_pos

theorem integral_exp_neg_sq_Ioi_zero :
    ∫ x in Ioi (0 : ℝ), exp (-x ^ 2) = √π / 2 := by
  have hfull := integral_exp_neg_sq
  have hsplit := integral_comp_abs (f := fun x : ℝ => exp (-x ^ 2))
  have hsame : (fun x : ℝ => exp (-|x| ^ 2)) = fun x => exp (-x ^ 2) := by
    ext x
    rw [sq_abs]
  rw [hsame] at hsplit
  linarith

noncomputable def ewaldErf (x : ℝ) : ℝ :=
  (2 / √π) * ∫ t in (0 : ℝ)..x, exp (-t ^ 2)

noncomputable def ewaldErfc (x : ℝ) : ℝ :=
  (2 / √π) * ∫ t in Ioi x, exp (-t ^ 2)

theorem ewald_split (x : ℝ) : ewaldErf x + ewaldErfc x = 1 := by
  unfold ewaldErf ewaldErfc
  have hint := integrable_exp_neg_sq
  have hadd :=
    integral_interval_add_Ioi (a := (0 : ℝ)) (b := x)
      (f := fun t : ℝ => exp (-t ^ 2)) hint.integrableOn hint.integrableOn
  have hhalf := integral_exp_neg_sq_Ioi_zero
  have hsum :
      (∫ t in (0 : ℝ)..x, exp (-t ^ 2)) + ∫ t in Ioi x, exp (-t ^ 2) = √π / 2 := by
    linarith
  have hpi : (2 / √π) * (√π / 2) = 1 := by
    field_simp
  calc
    ((2 / √π) * ∫ t in (0 : ℝ)..x, exp (-t ^ 2)) +
        ((2 / √π) * ∫ t in Ioi x, exp (-t ^ 2))
        = (2 / √π) *
            ((∫ t in (0 : ℝ)..x, exp (-t ^ 2)) + ∫ t in Ioi x, exp (-t ^ 2)) := by ring
    _ = (2 / √π) * (√π / 2) := by rw [hsum]
    _ = 1 := hpi

theorem ewaldErf_zero : ewaldErf 0 = 0 := by
  unfold ewaldErf
  simp

theorem ewaldErfc_zero : ewaldErfc 0 = 1 := by
  have h := ewald_split 0
  rw [ewaldErf_zero, zero_add] at h
  exact h

theorem ewaldErf_hasDerivAt (x : ℝ) :
    HasDerivAt ewaldErf ((2 / √π) * exp (-x ^ 2)) x := by
  unfold ewaldErf
  have hf : IntervalIntegrable (fun t : ℝ => exp (-t ^ 2)) volume 0 x :=
    integrable_exp_neg_sq.intervalIntegrable
  have hcont : Continuous (fun t : ℝ => exp (-t ^ 2)) :=
    continuous_exp.comp (continuous_neg.comp (continuous_pow 2))
  have hmeas : StronglyMeasurableAtFilter (fun t : ℝ => exp (-t ^ 2)) (nhds x) :=
    hcont.stronglyMeasurableAtFilter volume (nhds x)
  simpa using (integral_hasDerivAt_right hf hmeas hcont.continuousAt).const_mul (2 / √π)

theorem ewaldErf_deriv (x : ℝ) : deriv ewaldErf x = (2 / √π) * exp (-x ^ 2) :=
  (ewaldErf_hasDerivAt x).deriv

theorem ewaldErfc_tendsto_atTop : Tendsto ewaldErfc atTop (nhds 0) := by
  unfold ewaldErfc
  have htail :=
    tendsto_integral_Ioi_zero (μ := volume) (f := fun t : ℝ => exp (-t ^ 2))
      (l := atTop) (b := id) tendsto_id
  have hmul := Tendsto.const_mul (2 / √π) htail
  simpa [Function.comp] using hmul
