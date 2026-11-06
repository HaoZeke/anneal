import Mathlib.Analysis.SpecialFunctions.Erf

open Real Filter

/-- The Ewald split identity: the error function and the complementary
error function add to one. -/
theorem ewald_split (x : ℝ) : erf x + erfc x = 1 := by
  simpa using erf_add_erfc x

/-- The complementary error function is one at the origin, which is the
undamped limit of the real-space kernel. -/
theorem erfc_at_zero : erfc 0 = 1 := by
  simpa using erfc_zero

/-- The complementary error function vanishes at infinity, which is the
long-range limit of the real-space damping. -/
theorem erfc_vanishes_at_infinity : Tendsto erfc atTop (nhds 0) :=
  tendsto_erfc_atTop
