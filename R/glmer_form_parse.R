#' @export
glmer_form_parse <- function(formula, data=NULL, family = gaussian, sparseX = FALSE,
                  control = list(), start = NULL, verbose = 0L, nAGQ = 1L,
                  compDev = TRUE, subset, weights, na.action, offset,
                  contrasts = NULL, mustart, etastart, devFunOnly = FALSE,
                  tolPwrss = 1e-7, optimizer=c("bobyqa","Nelder_Mead"), ...)
{
  ## FIXME: does start= do anything? test & fix
  verbose <- as.integer(verbose)
  mf <- mc <- match.call()
  # extract family, call lmer for gaussian
  if (is.character(family))
    family <- get(family, mode = "function", envir = parent.frame(2))
  if( is.function(family)) family <- family()
  if (isTRUE(all.equal(family, gaussian()))) {
    mc[[1]] <- as.name("lmer")
    mc["family"] <- NULL            # to avoid an infinite loop
    return(eval(mc, parent.frame()))
  }
  if (family$family %in% c("quasibinomial", "quasipoisson", "quasi"))
    stop('"quasi" families cannot be used in glmer')
  
  checkArgs("glmer",sparseX,...)
  
  stopifnot(length(nAGQ <- as.integer(nAGQ)) == 1L,
            nAGQ >= 0L,
            nAGQ <= 25L)
  
  denv <- checkFormulaData(formula,data)
  mc$formula <- formula <- as.formula(formula,env=denv)    ## substitute evaluated version
  
  m <- match(c("data", "subset", "weights", "na.action", "offset",
               "mustart", "etastart"), names(mf), 0)
  mf <- mf[c(1, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1]] <- as.name("model.frame")
  fr.form <- subbars(formula) # substitute "|" for "+" -
  environment(fr.form) <- environment(formula)
  mf$formula <- fr.form
  fr <- eval(mf, parent.frame())
  # random-effects module
  reTrms <- mkReTrms(findbars(formula[[3]]), fr)
  if ((maxlevels <- max(unlist(lapply(reTrms$flist, nlevels)))) > nrow(fr))
    stop("number of levels of each grouping factor must be",
         "greater than or equal to number of obs")
  ## FIXME: adjust test for families with estimated scale;
  ##   useSc is not defined yet/not defined properly?
  ##  if (useSc && maxlevels == nrow(fr))
  ##          stop("number of levels of each grouping factor must be",
  ##                "greater than number of obs")
  
  ## fixed-effects model matrix X - remove random parts from formula:
  form <- formula
  form[[3]] <- if(is.null(nb <- nobars(form[[3]]))) 1 else nb
  X <- model.matrix(form, fr, contrasts)#, sparse = FALSE, row.names = FALSE) ## sparseX not yet
  p <- ncol(X)
  
  if ((rankX <- rankMatrix(X)) < p)
    stop(gettextf("rank of X = %d < ncol(X) = %d", rankX, p))

  out <- list(reTrms = reTrms, X = X)
  return(out)
}
