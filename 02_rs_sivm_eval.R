## ---------------------------
##
## 02_rs_sivm_eval.R
##
## This script showcases R routines useful for evaluating the results
## of vegetation stress detection with simplex volume maximization (SiVM).
##
## Author: Floris Hermanns
##
## Date Created: 2021-03-24
##
## ---------------------------

set.seed(1984)

library(corrplot)
library(ellipse)
library(Hmisc)

library(caret)
library(mctest)
library(betareg)
library(gamboostLSS)
library(ggplot2)

wdir <- 'path'
gsbvars <- read.csv(file.path(wdir, 'gsb_eval_vars.csv'))

#### Correlation analysis ####
gsbvars0 <- gsbvars
colnames(gsbvars0) = c('year', ':zeta', ':nu', ':Chl[REopt]', ':Car[REopt]', 'RENDVI', ':PRI[512]', 'CTR2', 'MCARI2', 'MSAVI2', 'WBI', ':f*minute(rho[950.6])', 'DEM', 'EM31', 'EM38', ':GR[DR]', ':GR[Th]')

cor_p <- rcorr(as.matrix(gsbvars0[ ,-1]))

png(height=1200, width=1200, pointsize=26, file='/home/hermanns/Nextcloud/Cloud/diss_docs/project01/plots/corr_subset.png', type = 'cairo')
par(xpd = TRUE)
corrplot.mixed(cor_p$r, cl.pos = 'n', upper = 'ellipse', tl.col = 'black', mar = c(0,0,0,1), outline = T, addgrid.col = NA, tl.pos = 'lt', tl.srt = 50, diag = 'l', p.mat = cor_p$P, sig.level = 0.0001)
# The p.mat is working but all correlations are highly significant so that no ellipses are crossed out
corrplot(cor_p$r, add = T, type = 'lower', method = 'number', col = 'black', diag = F, tl.pos = "n", cl.pos = "n", number.digits = 2, number.cex = .9)
dev.off()

#### Data preparation for inference ####
gsbvars$nu_stressed <- NULL # remove "stressed" response before regression
colnames(gsbvars) = c('year', 'zeta', 'chl_opt', 'car_opt', 'rendvi', 'pri512', 'ctr2', 'mcari2', 'msavi2', 'wbi', 'deriv950', 'dem', 'em31', 'em38', 'gr_dr', 'gr_th')

me <- c("center", "scale")
preProcv <- preProcess(gsbvars[ , 3:ncol(gsbvars)], method = me)
preProcv18 <- preProcess(gsbvars[gsbvars$year==2018, 3:ncol(gsbvars)], method = me)
preProcv19 <- preProcess(gsbvars[gsbvars$year==2019, 3:ncol(gsbvars)], method = me)

gsbvars_fm <- data.frame(gsbvars[ , 1:2],
                         as.matrix(predict(preProcv, gsbvars[ , 3:ncol(gsbvars)])),
                         int = rep(1, nrow(gsbvars)))
gsbvars_m18 <- data.frame(gsbvars[gsbvars$year==2018, 1:2],
                          as.matrix(predict(preProcv18, gsbvars[gsbvars$year==2018, 3:ncol(gsbvars)])),
                          int = rep(1, table(gsbvars$year==2018)[2]))
gsbvars_m19 <- data.frame(gsbvars[gsbvars$year==2019, 1:2],
                          as.matrix(predict(preProcv19, gsbvars[gsbvars$year==2019 , 3:ncol(gsbvars)])),
                          int = rep(1, table(gsbvars$year==2019)[2]))

#### MC testing ####
# remove year and int for usage of "." in regular formulas
gsbvars_clean <- gsbvars[, !(names(gsbvars) %in% c('year', 'int'))]
fm_beta = betareg(zeta ~ . | car_opt, data = gsbvars_clean)
coef(fm_beta)
plot(fm_beta) # diagnostic plots
plot(fm_beta, which = 5, type = "deviance", sub.caption= "")

# leverage testing ?
lp <- which(gleverage(fm_beta) > 50, arr.ind = F) # obs 171 with very high leverage
gleverage(fm_beta)[lp] # inspect leverage values
gsbvars_clean[lp,] # inspect data points -> no reason for exclusion

# Multicollinearity testing. Focus: condition number of regressor matrix and VIF values
# Check for presence of MC
omcdiag(fm_beta)
# Find location of MC with VIF
imcdiag(fm_beta)
# -> strong MC requires alternative model fitting approach

#### Fitting Boosted beta regression ####
ctrl <- boost_control(mstop = 300, nu = 0.05, trace = T) # model hyperparameters

# formula of linear base learners (bols)
form <- as.formula(zeta ~ bols(int, intercept = F) + bols(chl_opt, intercept = F) + bols(car_opt, intercept = F) + bols(rendvi, intercept = F) + bols(pri512, intercept = F) + bols(msavi2, intercept = F) + bols(wbi, intercept = F) + bols(ctr2, intercept = F) + bols(mcari2, intercept = F) + bols(deriv950, intercept = F) + bols(dem, intercept = F) + bols(em31, intercept = F) + bols(em38, intercept = F) + bols(gr_dr, intercept = F) + bols(gr_th, intercept = F)) # intercept false - using global intercept instead

fm0 <- gamboostLSS(form, families = BetaLSS(stabilization = "MAD"), data = gsbvars_fm, control = ctrl)
fm18 <- gamboostLSS(form, families = BetaLSS(stabilization = "MAD"), data = gsbvars_m18, control = ctrl)
fm19 <- gamboostLSS(form, families = BetaLSS(stabilization = "MAD"), data = gsbvars_m19, control = ctrl)

round(unlist(coef(fm0)[[1]]), 3) # mu coefs (expectation)
round(unlist(coef(fm0)[[2]]), 3) # phi coefs (precision)

round(unlist(coef(fm18)[[1]]), 3)
round(unlist(coef(fm18)[[2]]), 3)

round(unlist(coef(fm19)[[1]]), 3)
round(unlist(coef(fm19)[[2]]), 3)