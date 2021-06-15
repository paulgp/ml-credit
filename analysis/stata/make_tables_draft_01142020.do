
set trace on
set tracedepth 1


local data_path "../../data"
local output_path "../../output"

insheet using  "`data_path'/all_vals_race1_interestrate1.csv", comma clear names

keep race_dum_*  v1 default istestdata iscalibratedata sato orig_year_dum_* loan_purpose_dum_* document_type_dum_* investor_type_dum_*

save  "`output_path'/data_base.dta", replace

use "`output_path'/data_base.dta", clear
sum default if istestdata == "both"
local incidence = r(mean)
local n = string(r(N), "%9.0fc")
sum default if istestdata == "left_only" & iscalibratedata == "left_only"
local n_estimate = string(r(N), "%9.0fc")

insheet using "`output_path'/eval_output_race0_interestrate0.csv", comma clear names
keep if inlist(model, "Logit", "LogitNonLinear", "RandomForestIsotonic")
gen r2 = 1- (brier_score / (`incidence' * (1-`incidence')))
gen brier_adj = brier_score -  `incidence'*(1-`incidence')
local number_for_text = (brier_adj[3] - brier_adj[2])/brier_adj[2]
disp "`number_for_text'"
local number_for_text2 = (r2[3] - r2[2])/r2[2]
foreach x in roc precision brier_score r2 {
	if "`x'" == "brier_score" {
		local `x'_l0 = string(100*`x'[1], "%9.4f")		
		local `x'_lnl0 = string(100*`x'[2], "%9.4f")
		local `x'_lnl0_notrunc = 100*`x'[2]
		local `x'_rf0  = string(100*`x'[3], "%9.4f")
		local `x'_rf0_notrunc = 100*`x'[3]
		local `x' = string(100*(`x'[3] - `x'[2])/(`x'[2]-`incidence'*(1-(`incidence'))), "%9.2f")
		}
	else {
		local `x'_l0 = string(`x'[1], "%9.4f")
		local `x'_lnl0 = string(`x'[2], "%9.4f")
		local `x'_lnl0_notrunc = `x'[2]		
		local `x'_rf0  = string(`x'[3], "%9.4f")
		local `x'_rf0_notrunc = `x'[3]		
		local `x' = string(100*(`x'[3] - `x'[2])/`x'[2], "%9.2f")
		}
	local _`x'_rf = `x'[3]
	disp "``x''"		
}

local model_sub "p2c"
insheet using "`output_path'/PD_p1_p2c.csv", comma clear names
save "`output_path'/PD_p1_p2c.dta", replace
use "`output_path'/PD_p1_p2c.dta", clear
gen v1 = n - 1
merge 1:1 v1 using  "`output_path'/data_base.dta"
/* prtab default  p1 if istestdata == "both" */
/* Stata confirmed the same as python **/
/* r(precision) = .0589766390621662 */
/* r(roc_area) =  .8536567383605592 */
/* r(brier) =  .0071493045224705 */
* prtab default  p2c if istestdata == "both" , nograph
/* r(precision(AUC)) =  .0603793077170849 */
/* r(roc)   = .8534769435103243 */
/* r(brier) =  .0071676034399766 */
local r2_p2c = 1- (.0071676034399766 / (`incidence' * (1-`incidence')))
local brier_score_p2c = -.0071676034399766 
local roc_p2c = .8534769435103243
local precision_p2c = .0603793077170849
local r2_lnl0_`model_sub' = string(`r2_`model_sub'', "%9.4f")
local brier_score_lnl0_`model_sub' = string(`brier_score_`model_sub'', "%9.4f")
local roc_lnl0_`model_sub' = string(`roc_`model_sub'', "%9.4f")
local precision_lnl0_`model_sub' = string(`precision_`model_sub'', "%9.4f")
foreach x in roc precision brier_score r2 {
	if "`x'" == "brier_score" {
		local `x'_rf0_`model_sub'  =``x'_rf0'
		local `x'_lnl0_`model_sub'  = ``x'_lnl0_`model_sub''*100
		local `x'_`model_sub' = string(100*(`_`x'_rf' - ``x'_`model_sub'')/(``x'_`model_sub''-`incidence'*(1-`incidence')), "%9.2f")		
		}
	else {
		local `x'_rf0_`model_sub'  =``x'_rf0'
		local `x'_lnl0_`model_sub'  = ``x'_lnl0_`model_sub''
		local `x'_`model_sub' = string(100*(`_`x'_rf' - ``x'_`model_sub'')/``x'_`model_sub'', "%9.2f")
		}
}

local model_sub "int1"
insheet using "`output_path'/eval_output_race0_interestrate1.csv", comma clear names
keep if inlist(model, "LogitNonLinear", "RandomForestIsotonic")
gen r2 = 1- (brier_score / (`incidence' * (1-`incidence')))
foreach x in roc precision brier_score r2 {
	if "`x'" == "brier_score" {
		local `x'_lnl1 = string(`x'[1]*100, "%9.4f")
		local `x'_rf1  = string(`x'[2]*100, "%9.4f")
		local `x'_`model_sub' = string(100*(`x'[2] - `x'[1])/(`x'[1]-`incidence'*(1-`incidence')), "%9.2f")
		}
	else {
		local `x'_lnl1 = string(`x'[1], "%9.4f")
		local `x'_rf1  = string(`x'[2], "%9.4f")
		local `x'_`model_sub' = string(100*(`x'[2] - `x'[1])/`x'[1], "%9.2f")
		}
}


insheet using "`output_path'/eval_output_race1_interestrate0.csv", comma clear names
keep if inlist(model, "Logit", "LogitNonLinear", "RandomForestIsotonic")
gen r2 = 1- (brier_score / (`incidence' * (1-`incidence')))
foreach x in roc precision brier_score r2 {
	if "`x'" == "brier_score" {
		local `x'_l1 = string(`x'[1]*100, "%9.4f")		
		local `x'_lnl1 = string(`x'[2]*100, "%9.4f")
		local `x'_lnl1_notrunc = `x'[2]*100
		local `x'_rf1  = string(`x'[3]*100, "%9.4f")
		local `x'_rf1_notrunc  = `x'[3]*100
		}
	else {
		local `x'_l1 = string(`x'[1], "%9.4f")		
		local `x'_lnl1 = string(`x'[2], "%9.4f")
		local `x'_lnl1_notrunc = `x'[2]		
		local `x'_rf1  = string(`x'[3], "%9.4f")
		local `x'_rf1_notrunc  = `x'[3]
		}
	}


  
local model_sub "raceint"
use "`output_path'/PD_raceint.dta", clear
gen v1 = n - 1
merge 1:1 v1 using  "`output_path'/data_base.dta"
* prtab default  p_ri if istestdata == "both" , nograph
/* r(precision(AUC)) =  .05922856554389 */
/*   r(roc_area) =  .8543784748786331 */
/* r(brier) =  .0071492766269486 */
local r2_`model_sub' = 1- (.0071492766269486 / (`incidence' * (1-`incidence')))
local brier_score_`model_sub' = -.0071492766269486 
local roc_`model_sub' = .8543784748786331
local precision_`model_sub' = .05922856554389
local r2_lnl1_`model_sub' = string(`r2_`model_sub'', "%9.4f")
local brier_score_lnl1_`model_sub' = string(`brier_score_`model_sub'', "%9.4f")
local roc_lnl1_`model_sub' = string(`roc_`model_sub'', "%9.4f")
local precision_lnl1_`model_sub' = string(`precision_`model_sub'', "%9.4f")
foreach x in roc precision brier_score r2 {
	if "`x'" == "brier_score" {
		local `x'_rf1_`model_sub'  =``x'_rf1'
		local `x'_lnl1_`model_sub'  = ``x'_lnl1_`model_sub''*100		
		local `x'_`model_sub' = string(100*(`_`x'_rf' - ``x'_`model_sub'')/(``x'_`model_sub''-`incidence'*(1-`incidence')), "%9.2f")
		}
	else {
		local `x'_rf1_`model_sub'  =``x'_rf1'
		local `x'_lnl1_`model_sub'  = ``x'_lnl1_`model_sub''
		local `x'_`model_sub' = string(100*(`_`x'_rf' - ``x'_`model_sub'')/``x'_`model_sub'', "%9.2f")
		}
}


local model_sub "racint2"
use "`output_path'/PD_raceint.dta", clear
gen v1 = n - 1
merge 1:1 v1 using  "`output_path'/data_base.dta"
* prtab default  p_ri3 if istestdata == "both" , nograph
/* r(precision(AUC)) =  .0600473955273628 */
/*   r(roc_area) =  .854637412558611 */
/* r(brier) =  .0071457861792999 */
local r2_`model_sub' = 1- (.0071457861792999 / (`incidence' * (1-`incidence')))
local brier_score_`model_sub' = .0071457861792999 
local roc_`model_sub' = .854637412558611
local precision_`model_sub' = .0600473955273628
local r2_lnl1_`model_sub' = string(`r2_`model_sub'', "%9.4f")
local brier_score_lnl1_`model_sub' = string(`brier_score_`model_sub''*100, "%9.4f")
local roc_lnl1_`model_sub' = string(`roc_`model_sub'', "%9.4f")
local precision_lnl1_`model_sub' = string(`precision_`model_sub'', "%9.4f")
foreach x in roc precision brier_score r2 {
	local `x'_rf1_`model_sub' ``x'_rf1'
	local `x'_lnl1_`model_sub'  ``x'_lnl1_`model_sub''
	if "`x'" == "brier_score" {
		local `x'_`model_sub' = string(100*(`_`x'_rf' - ``x'_`model_sub'')/(``x'_`model_sub''-`incidence'*(1-`incidence')), "%9.2f")
		}
	else {
		local `x'_`model_sub' = string(100*(`_`x'_rf' - ``x'_`model_sub'')/``x'_`model_sub'', "%9.2f")
		}
}


/** each race run separately **/
local model_sub "racint3"
use "`output_path'/PD_racesepmodels.dta", clear
gen v1 = n - 1
merge 1:1 v1 using  "`output_path'/data_base.dta"
* prtab default  p_rd_sep if istestdata == "both" , nograph
/* r(precision(AUC)) =   .0600204430520535 */
/*            r(roc_area) =  .854263575638093  */
/*               r(brier) =  .0071468863297491 */
local r2_`model_sub' = 1- (.0071468863297491 / (`incidence' * (1-`incidence')))
local brier_score_`model_sub' = .0071468863297491
local roc_`model_sub' = .854263575638093
local precision_`model_sub' = .0600204430520535
local r2_lnl1_`model_sub' = string(`r2_`model_sub'', "%9.4f")
local r2_lnl1_`model_sub'_notr = `r2_`model_sub''
local brier_score_lnl1_`model_sub' = string(`brier_score_`model_sub''*100, "%9.4f")
local brier_score_lnl1_`model_sub'_notr = `brier_score_`model_sub''*100
local roc_lnl1_`model_sub' = string(`roc_`model_sub'', "%9.4f")
local roc_lnl1_`model_sub'_notr = `roc_`model_sub''
local precision_lnl1_`model_sub' = string(`precision_`model_sub'', "%9.4f")
local precision_lnl1_`model_sub'_notr = `precision_`model_sub''

foreach x in roc precision brier_score r2 {
	local `x'_rf1_`model_sub'_notr ``x'_rf1_notrunc'
	local `x'_lnl1_`model_sub'  ``x'_lnl1_`model_sub''
	
	if "`x'" == "brier_score" {
		local `x'_`model_sub' = string(100*(`_`x'_rf' - ``x'_`model_sub'')/(``x'_`model_sub''-`incidence'*(1-`incidence')), "%9.2f")
		}
	else {
		local `x'_`model_sub' = string(100*(`_`x'_rf' - ``x'_`model_sub'')/``x'_`model_sub'', "%9.2f")
		}
	}
/*
insheet using /Users/PSG24/Dropbox/ML-credit/Notes/PD_racesepmodels_linlogit.csv, comma clear names
save "`output_path'/PD_racesepmodels_linlogit.dta", replace
*/
use "`output_path'/PD_racesepmodels_linlogit.dta", clear
gen v1 = n - 1
merge 1:1 v1 using  "`output_path'/data_base.dta"
* prtab default  p_rd_sep if istestdata == "both" , nograph
/* r(AUC) =  .0588577836751938 */
/*            r(roc_area) =  .8498502654875348 */
/*                r(brier) =  .0071729362625874 */
local r2_`model_sub' = 1- (.0071729362625874 / (`incidence' * (1-`incidence')))
local brier_score_`model_sub' = .0071729362625874
local roc_`model_sub' = .8498502654875348
local precision_`model_sub' = .0588577836751938

local r2_l1_`model_sub' = string(`r2_`model_sub'', "%9.4f")
local r2_l1_`model_sub'_notr = `r2_`model_sub''
local brier_score_l1_`model_sub' = string(`brier_score_`model_sub''*100, "%9.4f")
local brier_score_l1_`model_sub'_notr = `brier_score_`model_sub''*100
local roc_l1_`model_sub' = string(`roc_`model_sub'', "%9.4f")
local roc_l1_`model_sub'_notr = `roc_`model_sub''
local precision_l1_`model_sub' = string(`precision_`model_sub'', "%9.4f")
local precision_l1_`model_sub'_notr = `precision_`model_sub''


capture file close fh
file open fh using "`output_path'/eval_table_draft_interestrate0_main_new.csv", write replace
local l_label "Logit"
local lnl_label "Nonlinear Logit"
local rf_label "Random Forest"
foreach model in l lnl rf {
	local	row "``model'_label' &"
	if "`model'" == "lnl" | "`model'" == "l" {
		foreach x in roc precision brier_score {
			local row "`row' & ``x'_`model'0' & ``x'_`model'1' & ``x'_`model'1_`model_sub'' "
			}
		local x r2
		local row "`row' & ``x'_`model'0' & ``x'_`model'1' & ``x'_`model'1_`model_sub'' "
		}
	else {
		foreach x in roc precision brier_score {
			local row "`row' & ``x'_`model'0' & \multicolumn{2}{c}{``x'_`model'1' } "
			}
		local x r2
		local row "`row' & ``x'_`model'0' &  \multicolumn{2}{c}{``x'_`model'1' } "
		}
	local row "`row' \\"
	disp "`row'"
	file write fh "`row'" _n
	}

file close fh


capture file close fh
file open fh using "`output_path'/decomp_interestrate0_main_new_a.tex", write replace
local roc_label "ROC-AUC"
local precision_label "Precision"
local brier_score_label "Brier Score"
local r2_label "\$ R^{2}\$"
foreach x in roc precision brier_score r2 {
	local max = ``x'_rf1_`model_sub'_notr'
	local min = ``x'_lnl0_notrunc'
	local norm_dist = `max' - `min'
	/* Race  Dummy first*/
	/* Race Interaction Second*/
	/* Tech Third*/
	disp "`min', ``x'_lnl1_notrunc', ``x'_lnl1_`model_sub'_notr', `max'"
	local step1 = 100*(``x'_lnl1_notrunc'-`min')/`norm_dist'
	local step1 = string(`step1', "%9.2f")
	local step2 = 100*(``x'_lnl1_`model_sub'_notr'-``x'_lnl1_notrunc')/`norm_dist'
	local step2 = string(`step2', "%9.2f")	
	local step3 = 100*(``x'_rf1_`model_sub'_notr'-``x'_lnl1_`model_sub'_notr')/`norm_dist'
	local step3 = string(`step3', "%9.2f")		
	local row "``x'_label' & `step1' & `step2' & `step3' \\"
	disp "`row'"
	file write fh "`row'" _n
	}
file close fh



capture file close fh
file open fh using "`output_path'/decomp_interestrate0_main_new_b.tex", write replace
local lnl_label "Nonlinear Logit"
local rf_label "Random Forest"
foreach x in roc precision brier_score r2 {
	local max = ``x'_rf1_`model_sub'_notr'
	local min = ``x'_lnl0_notrunc'
	local norm_dist = `max' - `min'
	/* Tech first*/
	/* Race Second*/
	local step1 = 100*(``x'_rf0_notrunc'-`min')/`norm_dist'
	local step1 = string(`step1', "%9.2f")
	local step2 = 100*(`max'-``x'_rf0_notrunc')/`norm_dist'
	local step2 = string(`step2', "%9.2f")	
	local row "``x'_label' & `step1' & `step2' \\"
	disp "`row'"
	file write fh "`row'" _n
	}

file close fh

/*** Dump raw data for Table 6 ***/
capture file close fh
file open fh using "`output_path'/decomp_interestrate0_main_new_a_rawdata.csv", write replace
foreach x in roc precision brier_score r2 {
	local max = ``x'_rf1_`model_sub'_notr'
	local min = ``x'_lnl0_notrunc'
	/* Race  Dummy first*/
	/* Race Interaction Second*/
	/* Tech Third*/
	disp "`min', ``x'_lnl1_notrunc', ``x'_lnl1_`model_sub'_notr', `max'"
	local row "`min', ``x'_lnl1_notrunc', ``x'_lnl1_`model_sub'_notr', `max'"
	disp "`row'"
	file write fh "`row'" _n
	}
file close fh


capture file close fh
file open fh using "`output_path'/decomp_interestrate0_main_new_b_rawdata.csv", write replace
foreach x in roc precision brier_score r2 {
	local max = ``x'_rf1_`model_sub'_notr'
	local min = ``x'_lnl0_notrunc'
	/* Tech first*/
	/* Race Second*/
	local row "`min', ``x'_rf0_notrunc', `max'"
	disp "`row'"
	file write fh "`row'" _n
	}

file close fh



/*** Model Subset **/
use "`output_path'/data_base.dta", clear
sum default if race_dum_unknown == 0 & istestdata == "both"
local incidence_model_dropunknown = r(mean)
local n_model_dropunknown = string(r(N), "%9.0fc")
sum default if race_dum_unknown == 0 & istestdata == "left_only" & iscalibratedata == "left_only"
local n_estimate_model_dropunknown = string(r(N), "%9.0fc")
sum default if (orig_year_dum_2009 == 1 | orig_year_dum_2010 == 1 |  orig_year_dum_2011 == 1 ) & istestdata == "both"
local incidence_no_fintech1 = r(mean)
local n_no_fintech1 = string(r(N), "%9.0fc")
sum default if (orig_year_dum_2009 == 1 | orig_year_dum_2010 == 1 |  orig_year_dum_2011 == 1 ) & istestdata == "left_only" & iscalibratedata == "left_only"
local n_estimate_no_fintech1 = string(r(N), "%9.0fc")
sum default if (loan_purpose_dum_1 == 1) & istestdata == "both"
local incidence_no_fintech2 = r(mean)
local n_no_fintech2 = string(r(N), "%9.0fc")
sum default if (loan_purpose_dum_1 == 1) & istestdata == "left_only" & iscalibratedata == "left_only"
local n_estimate_no_fintech2 = string(r(N), "%9.0fc")
sum default if (race_dum_asian==0 &  race_dum_black==0 & race_dum_nativeamalaskahawaii == 0 &  race_dum_unknown == 0 &  race_dum_whitehisp == 0) & istestdata == "both"
local incidence_white_only = r(mean)
local n_white_only = string(r(N), "%9.0fc")
sum default if (race_dum_asian==0 &  race_dum_black==0 & race_dum_nativeamalaskahawaii == 0 &  race_dum_unknown == 0 &  race_dum_whitehisp == 0) & istestdata == "left_only" & iscalibratedata == "left_only"
local n_estimate_white_only = string(r(N), "%9.0fc")
sum default if (document_type_dum_1 == 1 & (investor_type_dum_2 == 1 | investor_type_dum_3 == 1)) & istestdata == "both"
local incidence_gse_full = r(mean)
local n_gse_full = string(r(N), "%9.0fc")
sum default if (document_type_dum_1 == 1 & (investor_type_dum_2 == 1 | investor_type_dum_3 == 1)) & istestdata == "left_only" & iscalibratedata == "left_only"
local n_estimate_gse_full = string(r(N), "%9.0fc")
sum default if istestdata == "both"
local incidence_no_fico = r(mean)
local n_no_fico = string(r(N), "%9.0fc")
sum default if istestdata == "left_only" & iscalibratedata == "left_only"
local n_estimate_no_fico = string(r(N), "%9.0fc")

local n_p2c  `n_no_fico'
local n_estimate_p2c  `n_estimate_no_fico'
local n_int1  `n_no_fico'
local n_estimate_int1  `n_estimate_no_fico'


foreach model_sub in model_dropunknown no_fintech1 no_fintech2 white_only gse_full no_fico {
	insheet using "`output_path'/`model_sub'eval_output_race0_interestrate0.csv", comma clear names
	keep if inlist(model, "LogitNonLinear", "RandomForestIsotonic")
	gen r2 = 1- (brier_score / (`incidence_`model_sub'' * (1-`incidence_`model_sub'')))
	foreach x in roc precision brier_score r2 {
		if "`x'" == "brier_score" {
			local `x'_lnl0 = string(100*`x'[1], "%9.4f")
			local `x'_rf0  = string(100*`x'[2], "%9.4f")
			local `x' = string(100*(`x'[2] - `x'[1])/(`x'[1]-`incidence'*(1-(`incidence'))), "%9.2f")
			}
		else {
			local `x'_lnl0 = string(`x'[1], "%9.4f")
			local `x'_rf0  = string(`x'[2], "%9.4f")
			local `x'_`model_sub' = string(100*(`x'[2] - `x'[1])/`x'[1], "%9.2f")
			}
		}

	capture file close fh
	file open fh using "`output_path'/eval_table_draft_interestrate0_submodel`model_sub'.csv", write replace
	local lnl_label "Nonlinear Logit"
	local rf_label "Random Forest"
	foreach model in lnl rf {
		local	row "``model'_label' & "
		foreach x in roc precision brier_score r2 {
			local row "`row' & ``x'_`model'0'  "
			}
		local row "`row' \\"
		disp "`row'"
		file write fh "`row'" _n
		}
	file close fh
	}



/*** Race Prediction Models **/
local model "race_outcome"
use "`output_path'/data_base.dta", clear
gen race = (race_dum_black == 1 | race_dum_whitehisp == 1)
sum race if istestdata == "both"
local incidence_race = r(mean)

insheet using "`output_path'/eval_race_output_race0_interestrate0.csv", comma clear names
keep if inlist(model, "Logit", "LogitNonLinear", "RandomForestIsotonic")
gen r2 = 1- (brier_score / (`incidence_race' * (1-`incidence_race')))

foreach x in roc precision brier_score r2 {
	if "`x'" == "brier_score" {
		local `x'_l0 = string(10*`x'[1], "%9.4f")		
		local `x'_lnl0 = string(10*`x'[2], "%9.4f")
		local `x'_rf0  = string(10*`x'[3], "%9.4f")
		}
	else {
		local `x'_l0 = string(`x'[1], "%9.4f")
		local `x'_lnl0 = string(`x'[2], "%9.4f")
		local `x'_rf0  = string(`x'[3], "%9.4f")
		}
}



capture file close fh
file open fh using "`output_path'/eval_table_draft_race_interestrate0_main_new.tex", write replace
local l_label "Logit"
local lnl_label "Nonlinear Logit"
local rf_label "Random Forest"
foreach model in l lnl rf {
	local	row "``model'_label' &"
	foreach x in roc precision brier_score {
		local row "`row' & ``x'_`model'0'  "
		}
	local x r2
	local row "`row' & ``x'_`model'0' "
	local row "`row' \\"
	disp "`row'"
	file write fh "`row'" _n
	}

file close fh




/*** Default Comparisons ***/
insheet using "`output_path'/_race0_interestrate0.csv", comma clear names
save  "`output_path'/data_base_base.dta", replace

use  "`output_path'/data_base_base.dta", clear
merge 1:1 v1 using  "`output_path'/data_base.dta"
/* construct resolution / reliability */
foreach x in logitnonlinear randomforestisotonic {
	preserve
	fastxtile y_bin = `x', n(10)
	sum default
	count
	local n = r(n)
	collapse default (count) n_k = v1, by(`x')
	
	gen resolution =  (default - `incidence' )^2
	egen resolution2 = total(resolution*n_k)
	egen n = total(n_k)
	replace resolution2 = resolution2 /n

	gen reliability =  (`x' - default )^2
	egen reliability2 = total(reliability*n_k)
	replace reliability2 = reliability2 /n

	list resolution2 reliability2 in 1
	restore
}



gen race_bin = 0
replace race_bin = 1 if race_dum_asian == 1
replace race_bin = 2 if race_dum_black == 1
replace race_bin = 3 if race_dum_nativeamalaskahawaii == 1
replace race_bin = 4 if race_dum_unknown == 1
replace race_bin = 5 if race_dum_whitehisp == 1
capture label define race 0 "White Non-Hisp." 1 "Asian" 2 "Black" 3 "Native American, Alaska, Hawaiian" 4 "Unknown" 5 "White Hispanic"
label values race_bin race
gen logdiff = log(randomforestisotonic) - log(logitnonlinear)
gen diff = randomforestisotonic - logitnonlinear
gen pos_share = logdiff < 0
keep if inlist(race_bin,0,1,2,5)
collapse default pos_share (sd) diff , by(race_bin)
replace pos_share = pos_share - 0.5

local line_white_share "White Non-Hisp. & "
local line_asian_share "Asian & "
local line_black_share "Black & "
local line_hisp_share "White Hispanic & "
local line_white_sd "White Non-Hisp. & "
local line_asian_sd "Asian & "
local line_black_sd "Black & "
local line_hisp_sd "White Hispanic & "
local line_roc "ROC & "
local line_precision "Precision & "
local line_brier_score "Brier Score & "
local line_r2 "R\$^{2}\$ & "
local line_n "\# Obs. used for testing & "
local line_n_estimate "\# Obs. used for estimation & "

local output_white = string(pos_share[1]*100, "%9.2f")
local output_asian = string(pos_share[2]*100, "%9.2f")
local output_black = string(pos_share[3]*100, "%9.2f")
local output_hisp = string(pos_share[4]*100, "%9.2f")


foreach race in white asian black hisp {
	local line_`race'_share "`line_`race'_share' & `output_`race''"
	}

local output_white = string(diff[1], "%9.4f")
local output_asian = string(diff[2], "%9.4f")
local output_black = string(diff[3], "%9.4f")
local output_hisp = string(diff[4], "%9.4f")

foreach race in white asian black hisp {
	local line_`race'_sd "`line_`race'_sd' & `output_`race''"
	}


foreach x in roc precision brier_score r2 n n_estimate  {
	local line_`x' "`line_`x'' & ``x''"
	}


local case_model_dropunknown "race_dum_unknown == 0"
local case_no_fintech1 "orig_year_dum_2009 == 1 | orig_year_dum_2010 == 1 |  orig_year_dum_2011 == 1"
local case_no_fintech2
local case_white_only "(race_dum_asian==0 &  race_dum_black==0 & race_dum_nativeamalaskahawaii == 0 &  race_dum_unknown == 0 &  race_dum_whitehisp == 0)"
local case_gse_full "document_type_dum_1 == 1 & (investor_type_dum_2 == 1 | investor_type_dum_3 == 1)"
local case_no_fico "race_bin >= 0"

local model_sub "int1"
insheet using "`output_path'/_race0_interestrate1.csv", clear
merge 1:1 v1 using "`output_path'/data_base_base.dta", keepusing(randomforestisotonic) nogen
merge 1:1 v1 using  "`output_path'/data_base.dta"
gen race_bin = 0
replace race_bin = 1 if race_dum_asian == 1
replace race_bin = 2 if race_dum_black == 1
replace race_bin = 3 if race_dum_nativeamalaskahawaii == 1
replace race_bin = 4 if race_dum_unknown == 1
replace race_bin = 5 if race_dum_whitehisp == 1
capture label define race 0 "White Non-Hisp." 1 "Asian" 2 "Black" 3 "Native American, Alaska, Hawaiian" 4 "Unknown" 5 "White Hispanic"
label values race_bin race
gen logdiff = log(randomforestisotonic) - log(logitnonlinear)
gen pos_share = logdiff < 0
gen diff = randomforestisotonic - logitnonlinear
keep if inlist(race_bin,0,1,2,5)
collapse default pos_share (sd) diff , by(race_bin)
replace pos_share = pos_share - 0.5
local output_white_`model_sub' = string(pos_share[1]*100, "%9.2f")
local output_asian_`model_sub' = string(pos_share[2]*100, "%9.2f")
local output_black_`model_sub' = string(pos_share[3]*100, "%9.2f")
local output_hisp_`model_sub' = string(pos_share[4]*100, "%9.2f")
foreach race in white asian black hisp {
	local line_`race'_share "`line_`race'_share' & `output_`race'_`model_sub''"
	}
local output_white = string(diff[1], "%9.4f")
local output_asian = string(diff[2], "%9.4f")
local output_black = string(diff[3], "%9.4f")
local output_hisp = string(diff[4], "%9.4f")

foreach race in white asian black hisp {
	local line_`race'_sd "`line_`race'_sd' & `output_`race''"
	}

foreach x in roc precision brier_score r2 n n_estimate {
	local line_`x' "`line_`x'' & ``x'_`model_sub''"
	}

foreach model_sub in no_fico model_dropunknown no_fintech1  white_only gse_full  {
	insheet using  "`output_path'/`model_sub'_race0_interestrate0.csv", comma clear names
	save  "`output_path'/data_base_`model_sub'.dta", replace
	use "`output_path'/data_base_`model_sub'.dta", clear
	merge 1:1 v1 using  "`output_path'/data_base.dta"
	gen race_bin = 0
	replace race_bin = 1 if race_dum_asian == 1
	replace race_bin = 2 if race_dum_black == 1
	replace race_bin = 3 if race_dum_nativeamalaskahawaii == 1
	replace race_bin = 4 if race_dum_unknown == 1
	replace race_bin = 5 if race_dum_whitehisp == 1
	capture label define race 0 "White Non-Hisp." 1 "Asian" 2 "Black" 3 "Native American, Alaska, Hawaiian" 4 "Unknown" 5 "White Hispanic"
	label values race_bin race
	gen logdiff = log(randomforestisotonic) - log(logitnonlinear)
	gen pos_share = logdiff < 0
	gen diff = randomforestisotonic - logitnonlinear
	keep if inlist(race_bin,0,1,2,5)
	collapse default pos_share (sd) diff if `case_`model_sub'', by(race_bin)
	replace pos_share = pos_share - 0.5
	local output_white_`model_sub' = string(pos_share[1]*100, "%9.2f")
	local output_asian_`model_sub' = string(pos_share[2]*100, "%9.2f")
	local output_black_`model_sub' = string(pos_share[3]*100, "%9.2f")
	local output_hisp_`model_sub' = string(pos_share[4]*100, "%9.2f")

	foreach race in white asian black hisp {
		local line_`race'_share "`line_`race'_share' & `output_`race'_`model_sub''"
		}
	local output_white = string(diff[1], "%9.4f")
	local output_asian = string(diff[2], "%9.4f")
	local output_black = string(diff[3], "%9.4f")
	local output_hisp = string(diff[4], "%9.4f")

	foreach race in white asian black hisp {
		local line_`race'_sd "`line_`race'_sd' & `output_`race''"
	}

	foreach x in roc precision brier_score r2 n n_estimate {
		local line_`x' "`line_`x'' & ``x'_`model_sub''"
		}
	}

local model_sub "p2c"
use "`output_path'/PD_p1_p2c.dta", clear
gen v1 = n - 1
rename p2c logitnonlinear
merge 1:1 v1 using "`output_path'/data_base_base.dta", keepusing(randomforestisotonic) nogen
merge 1:1 v1 using  "`output_path'/data_base.dta"
gen race_bin = 0
replace race_bin = 1 if race_dum_asian == 1
replace race_bin = 2 if race_dum_black == 1
replace race_bin = 3 if race_dum_nativeamalaskahawaii == 1
replace race_bin = 4 if race_dum_unknown == 1
replace race_bin = 5 if race_dum_whitehisp == 1
capture label define race 0 "White Non-Hisp." 1 "Asian" 2 "Black" 3 "Native American, Alaska, Hawaiian" 4 "Unknown" 5 "White Hispanic"
label values race_bin race
gen logdiff = log(randomforestisotonic) - log(logitnonlinear)
gen pos_share = logdiff < 0
gen diff = randomforestisotonic - logitnonlinear
keep if inlist(race_bin,0,1,2,5)
collapse default pos_share (sd) diff , by(race_bin)
replace pos_share = pos_share - 0.5
local output_white_`model_sub' = string(pos_share[1]*100, "%9.2f")
local output_asian_`model_sub' = string(pos_share[2]*100, "%9.2f")
local output_black_`model_sub' = string(pos_share[3]*100, "%9.2f")
local output_hisp_`model_sub' = string(pos_share[4]*100, "%9.2f")
foreach race in white asian black hisp {
	local line_`race'_share "`line_`race'_share' & `output_`race'_`model_sub''"
	}
local output_white = string(diff[1], "%9.4f")
local output_asian = string(diff[2], "%9.4f")
local output_black = string(diff[3], "%9.4f")
local output_hisp = string(diff[4], "%9.4f")

foreach race in white asian black hisp {
	local line_`race'_sd "`line_`race'_sd' & `output_`race''"
	}

foreach x in roc precision brier_score r2 n n_estimate {
	local line_`x' "`line_`x'' & ``x'_`model_sub''"
	}



capture file close fh
file open fh using "`output_path'/eval_table_comparison.csv", write replace

foreach x in roc precision r2 {
	file write fh "`line_`x''\\" _n
	}
file write fh "\midrule" _n
file write fh "\multicolumn{7}{l}{Share who look more credit worthy in Random Forest vs. Logit Non-Linear} \\ "_n
file write fh "`line_white_share' \\" _n
file write fh "`line_asian_share' \\" _n
file write fh "`line_black_share' \\" _n
file write fh "`line_hisp_share' \\" _n
file write fh "\midrule" _n
file write fh "\multicolumn{7}{l}{Standard deviation of \$\Delta\$ probability of default from Random Forest vs. Logit Non-Linear} \\ "_n
file write fh "`line_white_sd' \\" _n
file write fh "`line_asian_sd' \\" _n
file write fh "`line_black_sd' \\" _n
file write fh "`line_hisp_sd' \\" _n
file write fh "\midrule" _n

foreach x in n n_estimate {
	file write fh "`line_`x''\\" _n
	}


file close fh

