---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: R
  language: R
  name: ir
---

```{code-cell} r
---
tags: []
vscode:
  languageId: r
---
pkgs<-c("microbenchmark", "tidyverse", "GenSA", "waldo")
invisible(sapply(pkgs, library, character.only = TRUE))
```

```{code-cell} r
---
vscode:
  languageId: r
---
stybtang <- function(xx)
{
  ##########################################################################
  #
  # STYBLINSKI-TANG FUNCTION
  #
  # Authors: Sonja Surjanovic, Simon Fraser University
  #          Derek Bingham, Simon Fraser University
  # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
  #
  # Copyright 2013. Derek Bingham, Simon Fraser University.
  #
  # THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
  # FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
  # derivative works, such modified software should be clearly marked.
  # Additionally, this program is free software; you can redistribute it
  # and/or modify it under the terms of the GNU General Public License as
  # published by the Free Software Foundation; version 2.0 of the License.
  # Accordingly, this program is distributed in the hope that it will be
  # useful, but WITHOUT ANY WARRANTY; without even the implied warranty
  # of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  # General Public License for more details.
  #
  # For function details and reference information, see:
  # http://www.sfu.ca/~ssurjano/
  #
  ##########################################################################
  #
  # INPUT:
  #
  # xx = c(x1, x2, ..., xd)
  #
  ##########################################################################

  sum <- sum(xx^4 - 16*xx^2 + 5*xx)

  y <- sum/2
  return(y)
}
```

Recall that the Styblinski-Tang is typically evaluated on a hypercube $x_i \in [-5, 5] \forall i \in [1,\ldots,d]$ and has

$$ f(x^*) = -39.16599d, \qquad x^*=(-2.903534,\ldots,-2.903534) $$

We will now check the values and times for both packages.

```{code-cell} r
---
vscode:
  languageId: r
---
check_out_sim <- function(values){
	optim_out <- values[[1]]
	gensa_out <- values[[2]]
	diff_val <- (abs(optim_out$value) - abs(gensa_out$value))
	diff_par <- (abs(optim_out$par) - abs(gensa_out$par))
	# print(diff_par)
	# print(diff_val)
	diffs <- (c(diff_val, diff_par) < 1e-4)
	# print(diffs)
	return (all(diffs))
}
```

```{code-cell} r
---
vscode:
  languageId: r
---
optim(c(0, 0), method = "SANN", stybtang) -> oout
GenSA::GenSA(par=c(0,0), fn=stybtang, lower=rep(-5,2), upper=rep(5,2)) -> genout
compare(oout$par, genout$par, tolerance = 1e-3)
compare(oout$value, genout$value, tolerance = 1e-3)
#check_out_sim(oout, genout)
```

```{code-cell} r
---
vscode:
  languageId: r
---
microbenchmark(optim(c(0, 0), method = "SANN", stybtang),
GenSA::GenSA(par=c(0,0), fn=stybtang, lower=rep(-5,2), upper=rep(5,2)),
check = check_out_sim) -> mbench
```

```{code-cell} r
---
vscode:
  languageId: r
---
mbench %>% pull(expr)-> exprCol
```

```{code-cell} r
---
vscode:
  languageId: r
---
levels(mbench$expr)[match("optim(c(0, 0), method = \"SANN\", stybtang)",
levels(mbench$expr))] <- "optim"
```

```{code-cell} r
---
vscode:
  languageId: r
---
levels(mbench$expr)[match("GenSA::GenSA(par = c(0, 0), fn = stybtang, lower = rep(-5, 2),      upper = rep(5, 2))",
levels(mbench$expr))] <- "GenSA"
```

```{code-cell} r
---
vscode:
  languageId: r
---
autoplot(mbench)+ggtitle("Time comparisons for 2D Styblinski-Tang")
ggsave(filename = "r_base_stybtang2d.pdf", device='pdf', dpi=300)
#width = width, height = height)
```

<!--
 Copyright 2023 rgoswami

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->
