

/* This do-file contains the SQL code for the data pull via the Federal Reserve's RADAR system
and does the cleaning/filtering of the sample in order to obtain the data for our analysis in "Predicatbly Unequal" 

NOTE: the McDash data can change over time (servicers get added/removed) so there is no guarantee that the data pulls 
below will generate exactly the same files as the ones we used for the analysis in the paper.
*/

/* Query for McDash "static" data:

SELECT hmda_id,close_qtr,term_nmon,orig_amt,prop_type,prop_state,prop_zip,appraisal_amt,occupancy_type,fico_orig,dti_ratio,mort_type,
loan_type_mcdash,int_type,purpose_type_mcdash,loan_src_type,doc_raw_type,source_type,seasoning_nmon,prod_type,document_type,units_no,
ltv_ratio,lien_type,jumbo_flg,termination_type,termination_dt,rand_no_mcd FROM lps.view_join_lps_chmda_match_static_2017 
WHERE (close_qtr >= '2009-03-01' AND close_qtr <= '2013-12-01' AND hmda_num_mcd_cand=1 AND mcd_num_hmda_cand=1)

-> call resulting file mcd_static.dta (use "compress" in Stata first, in order to reduce size) */

/* Query for McDash "dynamic" data:

SELECT hmda_id,as_of_mon_id,jumbo_type,investor_type,cur_int_rate,mba_stat FROM lps.view_join_lps_chmda_match_2017 WHERE 
(close_qtr >= '2009-03-01' AND close_qtr <= '2013-12-01' AND hmda_num_mcd_cand=1 AND mcd_num_hmda_cand=1)

-> call resulting file mcd_dynamic.dta */

/* Query for HMDA data:

SELECT hmda_id,action_qtr,applicant_ethnicity,applicant_income,applicant_race_1,applicant_sex,application_qtr,coapplicant_ethnicity,coapplicant_race_1,
coapplicant_sex,county_code,date_diff,edit_status,fips_code,hoepa_status,lien_status,loan_amount,loan_purpose,loan_type_hmda,msa_md,msa_md_census,
msa_md_census_medfaminc,msa_md_hud_medfaminc,occupancy,preapproval,property_type,purchaser_type,rate_spread,state_code 
FROM lps.view_join_lps_chmda_match_static_2017 WHERE (close_qtr >= '2009-03-01' AND close_qtr <= '2013-12-01' AND hmda_num_mcd_cand=1 AND mcd_num_hmda_cand=1)

-> call resulting file hmda.dta */

set more off

use hmda.dta, clear
compress

merge 1:1 hmda_id using mcd_static.dta, nogen
destring *, replace

merge 1:m hmda_id using mcd_dynamic.dta, nogen

bysort hmda_id (as_of_mon_id): gen obsnr = _n
keep if obsnr <= 36 - seasoning_nmon


g x = inlist(mba_stat , "9","F","L","R")
egen ever90dpd = max(x), by(hmda_id)
drop x

keep if obsnr == 6 - seasoning_nmon

gen orig_year = year(close_qtr)

egen msa_id = group(msa_md)

// Sample restrictions (using both info from McDash and HMDA):

keep if inlist(loan_type_mcdash,"C","D") 
keep if int_type == "1"
keep if inrange(ltv_ratio, 20,100) 
drop if lien_status == 2
drop if occupancy == 3 
drop if occupancy_type == "U" 
keep if prod_type=="10"
keep if property_type == 1 
keep if inlist(prop_type , "1","C") 
keep if mort_type==1
keep if inlist(term_nmon, 360, 180, 240, 120) 
keep if applicant_income <=500 
keep if orig_amt <= 10^6


// Define groups:
g race 		= "White non-hisp" 	if applicant_race_1 == 5 & applicant_ethnicity!=1
replace race 	= "White hisp" 	if applicant_race_1 == 5 & applicant_ethnicity==1
replace race 	= "Asian"	   	if applicant_race_1 == 2
replace race 	= "Black"	   	if applicant_race_1 == 3 
replace race	= "Native Am, Alaska, Hawaii" if inlist(applicant_race_1,1,4)
replace race	= "Unknown"		if inlist(applicant_race_1,6,7)

// Fico, LTV, Income bins
egen fico_bin = cut(fico_orig), at(280(20)870)
replace fico_bin = 0 if fico_bin == .
replace fico_bin =600 if inrange(fico_bin,1,599)
replace fico_bin =820 if fico_bin==840
egen ltv_bin = cut(ltv_ratio), at(20(5)105)
gen ltv_80 = ltv_ratio == 80
egen income_bin = cut(applicant_income), at(-25(25)525)

g coapplicant = inrange(coapplicant_sex,1,3) 

/* If want to use the data in Stata:
ds, has(type string)
foreach x in `r(varlist)' {
cap encode `x', gen(`x'_n)
}
drop prod_type_n int_type_n // no variation
*/

egen mortgage_rate = mean(cur_int_rate), by(close_qtr)
gen sato = cur_int_rate - mortgage_rate
drop if !inrange(sato,-1.5,1.5) // likely outliers/errors

// outsheet data year by year
forval i = 2009/2013 {
    outsheet if orig_year==`i' using sample`i'_clean_3yr.csv, comma replace
}

