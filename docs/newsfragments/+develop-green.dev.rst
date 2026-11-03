The ``PR title`` workflow runs ``cog verify`` on the title of each open pull request.
GitHub names the squash of a multi-commit pull request after its title.
A title that breaks the commit convention now fails before the merge.
The check reruns on every push to the pull request and on every edit of its title.
``a_rigid_chain_has_no_restart_among_its_stall_arms`` turns ``restart_on_stall`` on in its own configuration.
The recommended preset leaves it off.
A rigid chain still drops the restart from its stall arms.
``two_unreached_references_give_one_water_trajectory`` turns off the cycle reversal of ``Config::for_tip4p``.
Its chain then moves by translation and rotation alone and stalls inside the budget, as the test expects.
The reached-reference test keeps the full preset.
