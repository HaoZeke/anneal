``Config::call_floor`` declares the pair distance at or below which the objective refuses to evaluate, such as the 0.90 A floor of the GFN2 handle.
A trial with a pair inside the floor has that pair opened to 1.05 times the floor before the relaxation sees it, and every other pair is left alone.
A trial that cannot be opened is dropped without a charge, and so is a zero-step read that lands inside the floor.
The floor also keeps a species-free search's line searches off the floor and pushes a refused pair back out.
``Config::call_floor_elements`` names the atomic number of each point.
Each pair's floor is then ``element_call_floor``, 0.70 of the summed covalent radii and never below ``call_floor``.
Silicon pairs get 1.55 A and carbon pairs 1.06 A, and a pair whose scaled sum is below ``call_floor`` keeps ``call_floor``.
A floor that is not a positive distance, or an element list that does not name every point, is refused when the search starts.
Both fields are unset in every stock preset and are left out of the resolved configuration while unset.
