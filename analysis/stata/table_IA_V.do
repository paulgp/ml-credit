
// This file creates Table IA.V: Equilibrium Effects Under Alternative Approach


cd ~/Dropbox/ML-credit/


use  "code_Stata/output/data_base_base.dta", clear
merge 1:1 v1 using "code_Stata/output/data_base.dta", nogen

keep if (document_type_dum_1 == 1 & (investor_type_dum_2 == 1 | investor_type_dum_3 == 1)) // same as for eqm analysis

renvars race_dum_*, subst(race_dum_)
g race = "whitenonhisp"
foreach x in asian black nativeamalaskahawaii unknown whitehisp {
replace race = "`x'" if `x'==1
}

replace race="other" if inlist(race,"nativeamalaskahawaii","unknown")

global dim = 2

cap log close
local i = $dim

tab race

sum logitnonlinear
local a = r(min)
sum randomforestisotonic
count if randomforestisotonic<`a'

replace randomforestisotonic = `a' if randomforestisotonic<`a' // replace the few zeros in RF with the lowest PD from Logit

fp <logitnonlinear> , dimension($dim) : reg sato <logitnonlinear>   

matrix uu = e(fp_fp) // the powers chosen

predict r_logitnl // the predicted sato from the logitnonlinear

forval i = 1/$dim {
g double x`i' = logitnonlinear^uu[1,`i']
replace x`i' = ln(logitnonlinear) if uu[1,`i']==0 

replace logitnonlinear_`i' = randomforestisotonic^uu[1,`i']
replace logitnonlinear_`i' = ln(randomforestisotonic) if uu[1,`i']==0
}

predict r_rf // the predicted sato from the RF

// Table IA.V: Equilibrium Effects Under Alternative Approach
tabstat r_logitnl r_rf , stat(mean sd) by(race) format(%9.3f)
