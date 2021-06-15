

// This do-file creates Table 1, Table IA.II, and  Table IA.III of "Predictably Unequal?"
local data_path "../../data"
local output_path "../../output"

import delimited "`data_path'/all_vals_race1_interestrate1.csv", varnames(1) encoding(ISO-8859-1) clear

g race_dum_whitenonhisp = 1- race_dum_asian-race_dum_black-race_dum_nativeamalaskahawaii-race_dum_unknown-race_dum_whitehisp

g race = "asian"
foreach x in black nativeamalaskahawaii unknown whitehisp whitenonhisp {
replace race = "`x'" if race_dum_`x'==1
}
tab race, m

g orig_year=2013
forval i = 2009/2012 {
	replace orig_year = `i' if orig_year_dum_`i'==1
}

tab  orig_year race, row // fraction of unknown between 9.5% and 11.5%

replace applicant_income = . if applicant_income<0
replace fico_orig_fill   = . if fico_orig_fill==0 


// Descriptive statistics, Table 1 in the paper
preserve
	rename (fico_orig_fill applicant_income orig_amt cur_int_rate sato default) (FICO Income LoanAmt Rate SATO Default)
	replace LoanAmt = LoanAmt/1000
	replace Default = Default*100

	eststo clear
		estpost tabstat FICO Income LoanAmt Rate SATO Default ///
		, by(race) statistics(mean median sd N) nototal 

	esttab using "`output_path'/desc_allyears.tex", booktabs cells("FICO(fmt(%9.0f)) Income(fmt(%9.0f)) LoanAmt(fmt(%9.0f)) Rate(fmt(%9.2f)) SATO(fmt(%9.2f)) Default(fmt(%9.2f))" ) noobs nonumber nomtitle ///
	coeflabel(mean "Mean" p50 "Median" sd "SD") replace

restore

// Same for "equilibrium sample" (Table IA.II)
preserve
	keep if (investor_type_dum_2 ==1 | investor_type_dum_3 ==1) & document_type_dum_1 ==1

	rename (fico_orig_fill applicant_income orig_amt cur_int_rate sato default) (FICO Income LoanAmt Rate SATO Default)
	replace LoanAmt = LoanAmt/1000
	replace Default = Default*100

	eststo clear
		estpost tabstat FICO Income LoanAmt Rate SATO Default ///
		, by(race) statistics(mean median sd N) nototal 

	esttab using "`output_path'/desc_allyears_gsefd.tex", booktabs cells("FICO(fmt(%9.0f)) Income(fmt(%9.0f)) LoanAmt(fmt(%9.0f)) Rate(fmt(%9.2f)) SATO(fmt(%9.2f)) Default(fmt(%9.2f))" ) noobs nonumber nomtitle ///
	coeflabel(mean "Mean" p50 "Median" sd "SD") replace

restore


// Table IA.III. Residual Variation in SATO:

g prop_state = ""

foreach var of varlist prop_state_dum* {
		replace prop_state = `"`=substr("`var'",-2,2)'"' if `var'==1 
		}
replace prop_state = "wy" if prop_state=="" // omitted state -- wyoming
encode prop_state, gen(prop_state_nr)

drop prop_state_dum* prop_state

reghdfe sato ///
applicant_income income_miss ltv_ratio_fill fico_orig_miss fico_orig_fill orig_amt log_orig_amt occupancy_type_dum_* jumbo_flg_dum_n ///
investor_type_dum_* loan_purpose_dum_* coapplicant_dum_0 term_nmon_dum_* loan_type_mcdash_dum_* orig_year_dum_* document_type_dum_*, absorb(prop_state_nr) res(sato_res)

g eqm_sample     = document_type_dum_1 == 1 & (investor_type_dum_2==1 | investor_type_dum_3 ==1)
g not_eqm_sample = document_type_dum_1 != 1 | (investor_type_dum_2 !=1 & investor_type_dum_3 !=1)
tab not_eqm_sample eqm_sample


eststo clear
estpost tabstat sato_res sato, stat(sd) by(not_eqm_sample) nototal

label variable sato_res "SATO residual"
label variable sato "SATO"

esttab . using "`output_path'/tab_compare_resid_eqm.tex", booktabs cells("sato_res(fmt(%9.3f)) sato(fmt(%9.3f))") ///
nonumber noobs label nomtitle coeflabel(0 "Equilibrium Sample" 1 "Other") replace




