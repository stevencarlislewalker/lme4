#' @export
RZXbyhand <- function(object){
  Zt <- object$Zt
  X <- object$X
  Lambdat <- object$Lambdat

  Ut <- Lambdat %*% Zt
  Lt <- Cholesky(tcrossprod(Ut), Imult = 1)
  Q <- Ut %*% X

  out <- solve(Lt, Q, system = 'P')
  dimnames(out) <- NULL
  return(out)
}
