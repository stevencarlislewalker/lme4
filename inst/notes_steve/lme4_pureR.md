Intro to numerical approaches to mixed models: or simpler versions of lme4 in pure R
====================================================================================

Introduction
------------

The current development version of the `lme4` package is difficult for me to understand.  The complexity arises due to:

1. `C++` code
2. heavy use of reference classes, 
3. fill-reducing permutations for sparse Cholesky decompositions (which I gather are useful for speed, but difficult for me), 
4. and interactions between these things.

Here I decribe my work on writing a version in pure `R` that uses simpler linear algebra.

Assumptions
-----------

I think that all I assume is an understanding of three ideas:

1.  matrix multiplication (two matrices become one!)
2.  matrix decomposition (one matrix becomes two!)
3.  solving linear matrix equations
4.  that both of these things can be done on a computer, without having to think about the details of these computations

Basic numerical problem
-----------------------

One of the coolest things about linear mixed effects models as fitted by the `lmer` function, is that the fitting proceedure consists of iteratively solving a series of linear matrix equations.  A general such linear matrix equation posses the problem of finding a matrix $\mathbf W$, given,

$$\mathbf A \mathbf W = \mathbf C$$

where $\mathbf A $ is square and symmetric.  This can be done directly on a computer.  For example, in `R` we would use the `solve` command,


```r
set.seed(1)
(A <- matrix(rnorm(4), 2, 2))
```

```
##         [,1]    [,2]
## [1,] -0.6265 -0.8356
## [2,]  0.1836  1.5953
```

```r
(C <- matrix(rnorm(4), 2, 2))
```

```
##         [,1]   [,2]
## [1,]  0.3295 0.4874
## [2,] -0.8205 0.7383
```

```r
(W <- solve(A, C))
```

```
##         [,1]    [,2]
## [1,]  0.1891 -1.6486
## [2,] -0.5361  0.6526
```


This is fine if we only want to solve a few of these very small problems.  However, when fitting complex linear mixed effects models to very large data sets, this method becomes way to slow and so we need to take a less direct approach.  The approach that `lme4` takes is based on the Cholesky decomposition.

Cholesky decomposition
----------------------

Think of this decomposition as the analogue of square root for matrices.  Using this decomposition, we can solve $\mathbf A \mathbf W = \mathbf C $ in three steps (rather than the single step above).  This sounds worse at first:  three steps instead of a single step.  But it turns out that solving the following three problems is faster than the single problem above,

1. Find the Cholesky factor, $\mathbf L$ (i.e. matrix square root), of $\mathbf A$, $$\mathbf A = \mathbf L \mathbf L^\top$$
2. Solve the following equation for a matrix called $\mathbf T$, $$\mathbf L \mathbf T = \mathbf C$$
3. We can then solve for what we really want, $\mathbf W$, as, $$\mathbf W = \mathbf L^\top \mathbf W = \mathbf C$$

To understand the speed, consider this example,

```r
library(rbenchmark)
library(RcppEigen)
```

```
## Loading required package: Rcpp
```

```
## Loading required package: Matrix
```

```
## Loading required package: lattice
```

```
## Attaching package: 'Matrix'
```

```
## The following object(s) are masked from 'package:stats':
## 
## toeplitz
```

```r
library(rmv)

set.seed(3)

n <- 100
A <- rcov(n + 1, n)
C <- rnorm(n)

benchmark(explicit.chol = {
    L <- chol(A)
    T <- forwardsolve(t(L), C)
    backsolve(L, T)
}, solve = solve(A, C), solve.inv = solve(A) %*% C, solve.with.Matrix = solve(Matrix(A), 
    C), qr.solve = qr.solve(A, C), naive.explicit.chol = {
    L <- chol(A)
    T <- solve(t(L), C)
    solve(L, T)
}, solve.with.svd = with(svd(A), v %*% diag(1/d) %*% t(u) %*% C), replications = 100, 
    order = "relative", columns = c("test", "relative"))
```

```
##                  test relative
## 1       explicit.chol    1.000
## 2               solve    1.136
## 5            qr.solve    2.068
## 6 naive.explicit.chol    2.409
## 3           solve.inv    2.932
## 4   solve.with.Matrix    7.568
## 7      solve.with.svd   20.545
```



But what do these matrices mean?
--------------------------------

The trick is that each of these matrics have a complex structure underlying them.  Why do we need to worry about this structure?  Well, it is this structure that allows us to relate this simple matrtix equation to our statistical interpretation of it as a linear mixed model.

$$\mathbf A = \begin{bmatrix}
\mathbf P\left(\Lambda_\theta^\top\mathbf Z^\top\mathbf
      Z\Lambda_\theta+\mathbf I_q\right)\mathbf P^\top &
    \mathbf P\Lambda_\theta^\top\mathbf Z^\top\mathbf X\\
    \mathbf X^\top\mathbf Z\Lambda_\theta\mathbf P^\top & \mathbf X^\top\mathbf X
  \end{bmatrix}$$

$$\mathbf w = 
  \begin{bmatrix}
    \mathbf P\tilde{\mathbf u}\\
    \widehat{\mathbf\beta}_\theta
  \end{bmatrix}$$
  
  $$\mathbf c = 
  \begin{bmatrix}
    \mathbf P\Lambda_\theta^\top\mathbf Z^\top\mathbf y_{\text{obs}}\\
    \mathbf X^\top\mathbf y_{\text{obs}}
  \end{bmatrix}$$

To understand what these terms are I consider an example.  Although this example is meant to be as simple as possible, it is fairly lengthy and so a summary of the correspondence between the math symbols and the computational symbols follows.  I'm still going to use the formula parsing tools from `lme4` and I set the random number seed for replicability of the simulations below:


```r
library(lme4)
set.seed(1)
options(width = 100)
```


Simulating a **simple** null test data set:

```r
n <- 10  # sample size
p <- 2  # number of predictors

y <- rnorm(n)  # response
X <- matrix(rnorm(n * p), n, p)  # fixed effects design
f <- rep(c("a", "b"), c(5, 5))  # grouping factor for random effects

(df <- cbind(y, as.data.frame(X), f))  # data frame with these data
```

```
##          y       V1       V2 f
## 1  -0.6265  1.51178  0.91898 a
## 2   0.1836  0.38984  0.78214 a
## 3  -0.8356 -0.62124  0.07456 a
## 4   1.5953 -2.21470 -1.98935 a
## 5   0.3295  1.12493  0.61983 a
## 6  -0.8205 -0.04493 -0.05613 b
## 7   0.4874 -0.01619 -0.15580 b
## 8   0.7383  0.94384 -1.47075 b
## 9   0.5758  0.82122 -0.47815 b
## 10 -0.3054  0.59390  0.41794 b
```


Here is the model to be fitted to the data:

```r
fm <- y ~ V1 + V2 + (V1 + V2 | f)
```



```r
fm_df_glmer <- lme4:::glmer_form_parse(fm, df)
```


This `fm_df_glmer` object contains many objects required to fit the model to the data.  The first things are the design matrices---the transpose of the random effects design matrix, $\mathbf Z^\top$,

```r
(Zt <- fm_df_glmer$reTrms$Zt)
```

```
## 6 x 10 sparse Matrix of class "dgCMatrix"
##                                                                               
## a 1.000 1.0000  1.00000  1.000 1.0000  .        .        .       .      .     
## a 1.512 0.3898 -0.62124 -2.215 1.1249  .        .        .       .      .     
## a 0.919 0.7821  0.07456 -1.989 0.6198  .        .        .       .      .     
## b .     .       .        .     .       1.00000  1.00000  1.0000  1.0000 1.0000
## b .     .       .        .     .      -0.04493 -0.01619  0.9438  0.8212 0.5939
## b .     .       .        .     .      -0.05613 -0.15580 -1.4708 -0.4782 0.4179
```

and the fixed effects design matrix, $\mathbf X$,

```r
(X <- fm_df_glmer$X)
```

```
##    (Intercept)       V1       V2
## 1            1  1.51178  0.91898
## 2            1  0.38984  0.78214
## 3            1 -0.62124  0.07456
## 4            1 -2.21470 -1.98935
## 5            1  1.12493  0.61983
## 6            1 -0.04493 -0.05613
## 7            1 -0.01619 -0.15580
## 8            1  0.94384 -1.47075
## 9            1  0.82122 -0.47815
## 10           1  0.59390  0.41794
## attr(,"assign")
## [1] 0 1 2
```


There are several matrix products that both are used often in the fitting proceedure and are functions of the data only.  Therefore, we compute these upfront,

```r
XtX <- t(X) %*% X
Xty <- t(X) %*% y
ZtX <- Zt %*% X
Zty <- Zt %*% y
```


Another object in `gm_df_glmer` is `theta`, corresponding to $\theta$, which contains the initial values for the variance component parameters,

```r
(theta <- fm_df_glmer$reTrms$theta)
```

```
## [1] 1 0 0 1 0 1
```

The meaning of these variance component parameter is somewhat difficult to understand.  Essentially, they fill in certain elements of the $\Lambda_\theta^\top$ matrix,

```r
(Lambdat <- fm_df_glmer$reTrms$Lambdat)
```

```
## 6 x 6 sparse Matrix of class "dgCMatrix"
##                 
## [1,] 1 0 0 . . .
## [2,] . 1 0 . . .
## [3,] . . 1 . . .
## [4,] . . . 1 0 0
## [5,] . . . . 1 0
## [6,] . . . . . 1
```

From a theoretical point of view this `Lambdat` matrix is a square root of the covariance matrix of the random effects.  From a computational point of view, this `Lambdat` matrix is a little different from other `R` matrices, in that it is a **sparse** matrix of class `dgCMatrix`. See documentation on the `Matrix` package for more information on this class.  The key piece of information for our purposes is that there is a slot that contains the non-zero values,

```r
Lambdat@x
```

```
##  [1] 1 0 1 0 0 1 1 0 1 0 0 1
```

The information on where to put these numbers in the matrix itself is stored in other slots called `i` and `p`.  However, we don't need to really worry about how this works.  We can calculate `Lambdat@x` manually by using another element of `rt`,

```r
(Lind <- fm_df_glmer$reTrms$Lind)
```

```
##  [1] 1 2 4 3 5 6 1 2 4 3 5 6
```

This vector contains the indices required to convert `theta` to `Lambdat@x`,

```r
theta[Lind]
```

```
##  [1] 1 0 1 0 0 1 1 0 1 0 0 1
```

Note that this result is identical to `Lambdat@x`.  This is not a coincidence.  This manual updating will come in handy during the model fitting proceedure.

I'll save the remaining objects in `fm_df_glmer$reTrms` but put off describing them for now (mostly because I don't understand them myself right now!),

```r
(Gp <- fm_df_glmer$reTrms$Gp)
```

```
## [1] 0 6
```

```r
(lower <- fm_df_glmer$reTrms$lower)
```

```
## [1]    0 -Inf -Inf    0 -Inf    0
```

```r
(flist <- fm_df_glmer$reTrms$flist)
```

```
##    f
## 1  a
## 2  a
## 3  a
## 4  a
## 5  a
## 6  b
## 7  b
## 8  b
## 9  b
## 10 b
```

```r
(f <- fm_df_glmer$reTrms$cnms$f)
```

```
## [1] "(Intercept)" "V1"          "V2"
```


OK so that takes care of $\mathbf Z, \mathbf X$, and $\Lambda_\theta$.  Another simple one is $y_\mathrm{obs}$, which are the response data,


```r
y
```

```
##  [1] -0.6265  0.1836 -0.8356  1.5953  0.3295 -0.8205  0.4874  0.7383  0.5758 -0.3054
```


The $\mathbf w $ vector contains the estimate (or value at the current iterate) of the random, $\tilde{\mathbf u}$, and fixed, $\widehat{\beta}_\theta$, effect coefficients.

I've left The strangest and most obscure aspect of the problem for last.  The matrix $\mathbf P$ is a permutation matrix that doesn't change the statistical interpretation of the estimates but does improve the computational speed of the fitting proceedure.  One of the more difficult aspects of this matrix for me is the fact that it is apparently never actually computed.  Instead, $\mathbf P$ is better thought of as a shorthand notation for using a particularly efficient method for solving a system of equations.

Later...
--------

For convenience, Doug Bates also defines another matrix,

```r
(Ut <- Lambdat %*% Zt)
```

```
## 6 x 10 sparse Matrix of class "dgCMatrix"
##                                                                                  
## [1,] 1.000 1.0000  1.00000  1.000 1.0000  .        .        .       .      .     
## [2,] 1.512 0.3898 -0.62124 -2.215 1.1249  .        .        .       .      .     
## [3,] 0.919 0.7821  0.07456 -1.989 0.6198  .        .        .       .      .     
## [4,] .     .       .        .     .       1.00000  1.00000  1.0000  1.0000 1.0000
## [5,] .     .       .        .     .      -0.04493 -0.01619  0.9438  0.8212 0.5939
## [6,] .     .       .        .     .      -0.05613 -0.15580 -1.4708 -0.4782 0.4179
```

And I find that a couple more matrices on this theme are also useful,

```r
UtU <- UtUpI <- Ut %*% t(Ut)
diag(UtUpI) <- diag(UtUpI) + 1
```

The Cholesky decomposition of `UtUpI` is used alot too,

```r
(L <- Cholesky(UtUpI))
```

```
## 'MatrixFactorization' of Formal class 'dCHMsimpl' [package "Matrix"] with 10 slots
##   ..@ x       : num [1:12] 6 0.0318 0.0677 9.9877 0.6746 ...
##   ..@ p       : int [1:7] 0 3 5 6 9 11 12
##   ..@ i       : int [1:12] 0 1 2 1 2 2 3 4 5 4 ...
##   ..@ nz      : int [1:6] 3 2 1 3 2 1
##   ..@ nxt     : int [1:8] 1 2 3 4 5 6 -1 0
##   ..@ prv     : int [1:8] 7 0 1 2 3 4 5 -1
##   ..@ colcount: int [1:6] 3 2 1 3 2 1
##   ..@ perm    : int [1:6] 0 1 2 3 4 5
##   ..@ type    : int [1:4] 2 0 0 1
##   ..@ Dim     : int [1:2] 6 6
```

The `L` object is a **sparse** Cholesky factor of class `dCHMsimpl`.

Here's something I don't get yet...updating.  Doug Bates uses lines like this,

```r
(L <- update(L, Ut, mult = 1))
```

```
## 'MatrixFactorization' of Formal class 'dCHMsimpl' [package "Matrix"] with 10 slots
##   ..@ x       : num [1:12] 6 0.0318 0.0677 9.9877 0.6746 ...
##   ..@ p       : int [1:7] 0 3 5 6 9 11 12
##   ..@ i       : int [1:12] 0 1 2 1 2 2 3 4 5 4 ...
##   ..@ nz      : int [1:6] 3 2 1 3 2 1
##   ..@ nxt     : int [1:8] 1 2 3 4 5 6 -1 0
##   ..@ prv     : int [1:8] 7 0 1 2 3 4 5 -1
##   ..@ colcount: int [1:6] 3 2 1 3 2 1
##   ..@ perm    : int [1:6] 0 1 2 3 4 5
##   ..@ type    : int [1:4] 2 0 0 1
##   ..@ Dim     : int [1:2] 6 6
```

In other words the `L` matrix is *updated*, but I'm perplexed because its no different than the original `L` in this case at least.  Maybe it'll make more sense later.
