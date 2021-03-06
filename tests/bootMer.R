library(lme4)

mySumm <- function(.) { s <- sigma(.)
                        c(beta =getME(., "beta"),
                          sigma = s, sig01 = unname(s * getME(., "theta"))) }
fm1 <- lmer(Yield ~ 1|Batch, Dyestuff)
boo01 <- bootMer(fm1, mySumm, nsim = 10)
boo02 <- bootMer(fm1, mySumm, nsim = 10, use.u = TRUE)

## boo02 <- bootMer(fm1, mySumm, nsim = 500, use.u = TRUE)
## library(boot)
##  boot.ci(boo02,index=2,type="perc")

fm2 <- lmer(angle ~ recipe * temperature + (1|recipe:replicate), cake)
boo03 <- bootMer(fm2, mySumm, nsim = 10)
boo04 <- bootMer(fm2, mySumm, nsim = 10, use.u = TRUE)

gm1 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
             data = cbpp, family = binomial)
boo05 <- bootMer(gm1, mySumm, nsim = 10)
boo06 <- bootMer(gm1, mySumm, nsim = 10, use.u = TRUE)

cbpp$obs <- factor(seq(nrow(cbpp)))
gm2 <- glmer(cbind(incidence, size - incidence) ~ period +
             (1 | herd) +  (1|obs),
             family = binomial, data = cbpp)
boo03 <- bootMer(gm2, mySumm, nsim = 10)
boo03 <- bootMer(gm2, mySumm, nsim = 10, use.u = TRUE)
