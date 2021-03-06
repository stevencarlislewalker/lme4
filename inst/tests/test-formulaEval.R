library("testthat")
library("lme4")

context("data= argument and formula evaluation")

test_that("glmerForm", {
    set.seed(101)

    x <- rbinom(10, 1, 1/2)
    y <- rnorm(10)
    z <- rnorm(10)
    r <- sample(1:3, size=10, replace=TRUE)
    d <- data.frame(x,y,z,r)

    F <- "z"
    rF <- "(1|r)"
    modStr <- (paste("x ~", "y +", F, "+", rF))
    modForm <- as.formula(modStr)

    ## formulas have environments associated, but character vectors don't
    
    ## data argument not specified:
    ## should** work, but documentation warns against it
    m_nodata.0 <- glmer( x ~ y + z + (1|r) , family="binomial")
    m_nodata.1 <- glmer( as.formula(modStr) , family="binomial")
    m_nodata.2 <- glmer( modForm , family="binomial")
    m_nodata.3 <- glmer( modStr , family="binomial")
    m_nodata.4 <- glmer( "x ~ y + z + (1|r)" , family="binomial")

    ## data argument specified
    m_data.0 <- glmer( x ~ y + z + (1|r) , data=d, family="binomial")
    m_data.1 <- glmer( as.formula(modStr) , data=d, family="binomial")
    m_data.2 <- glmer( modForm , data=d, family="binomial")
    m_data.3 <- glmer( modStr , data=d, family="binomial")
    m_data.4 <- glmer( "x ~ y + z + (1|r)" , data=d, family="binomial")

    ff <- function() {
        d2 <- data.frame(x,y,z,r)
        glmer( x ~ y + z + (1|r), data=d2, family="binomial")
    }
    m_data.5 <- ff()
    
    ## apply drop1 to all of these ...
    m_nodata_List <- list(m_nodata.0,m_nodata.1,m_nodata.2,m_nodata.3,m_nodata.4)
    d_nodata_List <- lapply(m_nodata_List,drop1)

    m_data_List <- list(m_data.0,m_data.1,m_data.2,m_data.3,m_data.4,m_data.5)
    d_data_List <- list()

    OKvals <- c(1:3,6)  ## skip 4,5
    
    for (i in seq_along(m_data_List[OKvals])) {
        drop1((m_data_List[OKvals])[[i]])
    }

    ## lapply(m_data_List,
    ## function(x) { ls(envir=environment(formula(getCall(x)))) })

    d_data_List <- lapply(m_data_List[OKvals],drop1)
    expect_error(lapply(m_data_List[4],drop1))
    expect_error(lapply(m_data_List[5],drop1))
    ## d_data_List <- lapply(m_data_List,drop1,evalhack="parent")  ## fails on element 1
    ## d_data_List <- lapply(m_data_List,drop1,evalhack="formulaenv")  ## fails on element 4
    ## d_data_List <- lapply(m_data_List,drop1,evalhack="nulldata")  ## succeeds
    ## drop1(m_data.5,evalhack="parent") ## 'd2' not found
    ## drop1(m_data.5,evalhack="nulldata") ## 'x' not found (d2 is in environment ...)
    ## should we try to make update smarter ... ??

    ## test equivalence of (i vs i+1) for all models, all drop1() results
    for (i in 1:(length(m_nodata_List)-1)) {
        expect_equivalent(m_nodata_List[[i]],m_nodata_List[[i+1]])
        expect_equivalent(d_nodata_List[[i]],d_nodata_List[[i+1]])
    }

    expect_equivalent(m_nodata_List[[1]],m_data_List[[1]])
    expect_equivalent(d_nodata_List[[1]],d_data_List[[1]])
    
    for (i in 1:(length(m_data_List[OKvals])-1)) {
        expect_equivalent(m_data_List[OKvals][[i]],m_data_List[OKvals][[i+1]])
        expect_equivalent(d_data_List[[i]],d_data_List[[i+1]])
    }
})


test_that("lmerForm", {

    set.seed(101)

    x <- rnorm(10)
    y <- rnorm(10)
    z <- rnorm(10)
    r <- sample(1:3, size=10, replace=TRUE)
    d <- data.frame(x,y,z,r)

    ## example from Joehanes Roeby
    m2 <- lmer(x ~ y + z + (1|r), data=d)
    ff <- function() {
        m1 <- lmer(x ~ y + z + (1|r), data=d)
        return(anova(m1))
    }

    ff1 <- Reaction ~ Days + (Days|Subject)
    fm1 <- lmer(ff1, sleepstudy)
    fun <- function () {
        ff1 <- Reaction ~ Days + (Days|Subject)
        fm1 <- lmer(ff1, sleepstudy)
        return (anova(fm1))
    }
    anova(m2)
    ff()
    expect_equal(anova(m2),ff())
    anova(fm1)
    fun()
    expect_equal(anova(fm1),fun())
})

