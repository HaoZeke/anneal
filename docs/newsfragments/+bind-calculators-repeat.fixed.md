Repeating `bindCalculators` with an argument of 0 or less returns the stored
group when it names the same split. Before, the repeat compared the raw
argument with the resolved size and returned index -1.
