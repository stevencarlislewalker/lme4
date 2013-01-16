#' @export
VtrUtrbyhand <- function(object){
  Zt <- object$Zt
  X <- object$X
  Lambdat <- object$Lambdat
  mu <- object$mu
  y <- object$y
  u <- object$u
  
  Vtr <- as.vector(t(X) %*% (y - mu))
  Utr <- as.vector((Lambdat %*% Zt %*% (y - mu))) - u
  
  list(Vtr = Vtr, Utr = Utr)
}


#RZXbyhand <- function(object){
#  Zt <- object$Zt
#  X <- object$X
#  Lambdat <- object$Lambdat
#  mu <- object$mu
#  y <- object$y
#  u <- object$u
#  N <- object$N
#  
#  LambdatZt <- Lambdat %*% Zt
#  #if(is.null(mu)) H <- diag(0.1875, ncol(Zt), ncol(Zt))
#  if(is.null(mu)) H <- diag(1, ncol(Zt), ncol(Zt))
#  else H <- N * diag(mu * (1 - mu), length(mu), length(mu))
#  H <- as(H, 'dgCMatrix')
#  LambdatZtH <- LambdatZt %*% H
#  
#  UtU <- tcrossprod(LambdatZtH, LambdatZt)
#  UtV <- LambdatZtH %*% X
#  VtV <- crossprod(X, crossprod(H, X))
#  
#  Lt <- Cholesky(UtU, LDL = FALSE, Imult = 1)
#  RZX <- solve(Lt, UtV, system = 'L')
#  RX <- Cholesky(as(VtV-crossprod(RZX), 'dgCMatrix'))
#  
#  dimnames(RZX) <- dimnames(RX) <- dimnames(Lt) <- NULL
#  
#  out <- list(UtV = UtV, RZX = RZX)
#  
#  if(!any(is.null(mu), is.null(y), is.null(u))){
#  
#    wtres <- (y - N * mu)*(N * (mu * (1 - mu)))
#    Vtr <- as.vector(t(X) %*% wtres)
#    Utr <- as.vector(Lambdat %*% Zt %*% wtres) - u
#    
#    out <- c(out, list(Vtr = Vtr, Utr = Utr))
#  
#  }
    
#  print('RZX by hand:')
#  return(out$RZX)
#
#}


#' @export
RZXbyhand <- function(pp, resp, includeH = TRUE){
  
  LambdatZt <- pp$Lambdat %*% pp$Zt
  
  if(includeH) Winv <- diag(resp$n * resp$mu * (1 - resp$mu), length(resp$mu), length(resp$mu))
  else Winv <- diag(1, ncol(LambdatZt), ncol(LambdatZt))
  
  Winv <- as(Winv, 'dgCMatrix')
  LambdatZtWinv <- LambdatZt %*% Winv
  
  UtU <- tcrossprod(LambdatZtWinv, LambdatZt)
  UtV <- LambdatZtWinv %*% pp$X
  VtV <- crossprod(pp$X, crossprod(Winv, pp$X))
  
  Lt <- Cholesky(UtU, LDL = FALSE, Imult = 1)
  RZX <- solve(Lt, UtV, system = 'L')
  RX <- Cholesky(as(VtV-crossprod(RZX), 'dgCMatrix'))
  
  dimnames(RZX) <- dimnames(RX) <- dimnames(Lt) <- NULL

  print('R_ZX by hand:')
  raw_matrix_print(RZX)
  
  #print('R_X by hand:')
  #print(RX)
  
  #print('L_t by hand:')
  #print(Lt)
  
  #print('U_t_V by hand:')
  #raw_matrix_print(UtV)
  
  return(RZX)
  
  ### DISREGARD BELOW THIS LINE
  
  #out <- list(UtV = UtV, RZX = RZX)
  
  #if(!any(is.null(mu), is.null(y), is.null(u))){
    
  #  wtres <- (y - N * mu)*(N * (mu * (1 - mu)))
  #  Vtr <- as.vector(t(X) %*% wtres)
  #  Utr <- as.vector(Lambdat %*% Zt %*% wtres) - u
    
  #  out <- c(out, list(Vtr = Vtr, Utr = Utr))
    
  #}
  
  #print('RZX by hand:')
  #return(out$RZX)
  
}

#' Convert reference class objects to lists
#' 
#' Store the current return values of the fields of a reference class object in the
#' elements of a list object.
#' 
#' @param obj An r5 object.
#' @return A list.
#' @export
r5_to_list <- function(obj){
  
  # get the names of the fields in the r5 object, obj
  obj_names <- ls(obj)
  
  # store the current value of each field in each of the elements of a list object, out
  out <- list()
  for(i in obj_names) out[[i]] <- obj[[i]]
  
  # set the element names to be the field names
  names(out) <- obj_names
  
  return(out)
}


#' @export
raw_matrix_print <- function(x) for(i in 1:nrow(x)) cat(x[i,], '\n')
