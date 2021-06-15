

local path "~/Dropbox/ML-credit/code_Python/code_2019_07_05_JFrev_PGP/output/"

forvalues i = 0/99 {
	insheet using "`path'/bootstrap`i'eval_output_race0_interestrate0.csv", comma clear names
	gen eval_num = `i'
	capture append using `tmp'	
	tempfile tmp
	save `tmp'
	}
local incidence = .0074066 
gen r2 = 1- (brier_score / (`incidence' * (1-`incidence')))
keep if model == "RandomForestIsotonic" | model == "LogitNonLinear"

reshape wide roc precision brier_score r2, i(eval_num) j(model) string

foreach x in roc precision brier_score r2 { 
	sum `x'LogitNonLinear, d
	local `x'NLp5 = r(p5)
	local `x'NLp95 = r(p95)
	sum `x'RandomForestIsotonic, d
	local `x'RFp5 = r(p5)
	local `x'RFp95 = r(p95)
	kdensity `x'LogitNonLinear, gen(x d)
	kdensity `x'RandomForestIsotonic, gen(x2 d2) 
	twoway (line d x) (line d2 x2), name(`x', replace) legend(rows(2))
	drop d x d2 x2
}



foreach x in roc precision brier_score r2 {
	disp "`x' Non-Linear Logit CI: [``x'NLp5', ``x'NLp95']"
	disp "`x' RF CI: [``x'RFp5', ``x'RFp95']"
	}
