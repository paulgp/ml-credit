

use "/Users/Paul/Dropbox/ML-credit/sato_06112018/all_vals_race0_interestrate1_nomsa"  if iscalibratedata == "left_only"   & istestdata == "both", clear

gen rf_round = round(randomforestisotonic, .0001)
gen nl_round = round(logitnonlinear, .0001)
gen logit_round = round(logit, .0001)

gen brier_rf = (default - rf_round)^2
gen brier_nl = (default - nl_round)^2
gen brier_logit = (default - logit_round)^2

sum default
local mean = r(mean)
egen rf_round_mean = mean(default), by(rf_round)
egen nl_round_mean = mean(default), by(nl_round)
egen logit_round_mean = mean(default), by(logit_round)

gen reliability_rf = (rf_round - rf_round_mean)^2
gen reliability_nl = (nl_round - nl_round_mean)^2
gen reliability_logit = (logit_round - logit_round_mean)^2

gen resolution_rf = (rf_round_mean - `mean')^2
gen resolution_nl = (nl_round_mean - `mean')^2
gen resolution_logit = (logit_round_mean - `mean')^2

gen uncertainty = `mean'* (1-`mean')

tabstat brier_rf brier_nl brier_logit, stat(mean)
tabstat reliability_*, stat(mean)
tabstat resolution_*, stat(mean)
disp `mean'* (1-`mean')
 

/*** Revision**/


use "/Users/PSG24/Dropbox/ML-credit/code_Stata/output/data_base.dta", clear
merge 1:1 v1 using  "/Users/PSG24/Dropbox/ML-credit/code_Stata/output/data_base_base.dta"
keep if iscalibratedata == "left_only"   & istestdata == "both"

gen rf_round = round(randomforestisotonic, .0001)
gen nl_round = round(logitnonlinear, .0001)
gen logit_round = round(logit, .0001)

gen brier_rf = (default - rf_round)^2
gen brier_nl = (default - nl_round)^2
gen brier_logit = (default - logit_round)^2

sum default
local mean = r(mean)
egen rf_round_mean = mean(default), by(rf_round)
egen nl_round_mean = mean(default), by(nl_round)
egen logit_round_mean = mean(default), by(logit_round)

gen reliability_rf = (rf_round - rf_round_mean)^2
gen reliability_nl = (nl_round - nl_round_mean)^2
gen reliability_logit = (logit_round - logit_round_mean)^2

gen resolution_rf = (rf_round_mean - `mean')^2
gen resolution_nl = (nl_round_mean - `mean')^2
gen resolution_logit = (logit_round_mean - `mean')^2

gen uncertainty = `mean'* (1-`mean')

tabstat brier_rf brier_nl brier_logit, stat(mean)
tabstat reliability_*, stat(mean)
tabstat resolution_*, stat(mean)
disp `mean'* (1-`mean')

test `mean'* (1-`mean') + 


/** Subset region to drop right tail **/
use "/Users/PSG24/Dropbox/ML-credit/code_Stata/output/data_base.dta", clear
merge 1:1 v1 using  "/Users/PSG24/Dropbox/ML-credit/code_Stata/output/data_base_base.dta"
keep if iscalibratedata == "left_only"   & istestdata == "both"

gen rf_round = round(randomforestisotonic, .0001) if randomforestisotonic < .15
gen nl_round = round(logitnonlinear, .0001) if logitnonlinear < .15
gen logit_round = round(logit, .0001) if logit < .15

gen brier_rf = (default - rf_round)^2
gen brier_nl = (default - nl_round)^2
gen brier_logit = (default - logit_round)^2


foreach x in randomforestisotonic logitnonlinear logit {
	sum default if `x' < .15
	local `x'_mean = r(mean) 
}

egen rf_round_mean = mean(default) if rf_round !=., by(rf_round)
egen nl_round_mean = mean(default) if nl_round !=. , by(nl_round)
egen logit_round_mean = mean(default) if logit_round !=., by(logit_round)

gen reliability_rf = (rf_round - rf_round_mean)^2
gen reliability_nl = (nl_round - nl_round_mean)^2
gen reliability_logit = (logit_round - logit_round_mean)^2

gen resolution_rf = (rf_round_mean - `randomforestisotonic_mean')^2
gen resolution_nl = (nl_round_mean - `logitnonlinear_mean')^2
gen resolution_logit = (logit_round_mean - `logit_mean')^2

gen uncertainty = `mean'* (1-`mean')

tabstat brier_rf brier_nl brier_logit, stat(mean)
tabstat reliability_*, stat(mean)
tabstat resolution_*, stat(mean)
disp `randomforestisotonic_mean'* (1-`randomforestisotonic_mean')
disp `logitnonlinear_mean'* (1-`logitnonlinear_mean')
disp `logit_mean'* (1-`logit_mean')


