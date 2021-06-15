

set rmsg on, perm
set matsize 10000

local data_path "../../data"
local output_path "../../output"
insheet using `data_path'/all_vals_race1_interestrate1.csv, comma clear names

gen n=_n

gen sample = istestdata != "both" & iscalibratedata != "both"


** Fico, LTV, Income
egen fico_bin = cut(fico_orig_fill), at(280(20)870)
replace fico_bin = 0 if fico_orig_miss == 1
egen ltv_bin = cut(ltv_ratio), at(20(5)105)
gen ltv_80 = ltv_ratio == 80
egen income_bin = cut(applicant_income), at(-25(25)525)
replace fico_bin =600 if inrange(fico_bin,1,599)
replace fico_bin =820 if fico_bin==840

 tab income_bin, gen(income_bin_dum)
 tab ltv_bin, gen(ltv_bin_dum)
 tab fico_bin, gen(fico_bin_dum)
 sum *bin_dum*  if sample == 1

// basic logit specification
logit default orig_amt log_orig_amt occupancy_type_dum_1 occupancy_type_dum_2  jumbo_flg_dum_n investor_type_dum_1 investor_type_dum_2 investor_type_dum_3 ///
loan_purpose_dum_1 loan_purpose_dum_2 coapplicant_dum_0 term_nmon_dum_120 term_nmon_dum_180 term_nmon_dum_240 loan_type_mcdash_dum_c orig_year_dum_* prop_state_dum* ///
document_type_dum_1 document_type_dum_2 document_type_dum_3 income_bin_dum* fico_bin_dum* ltv_bin_dum* ltv_80  if sample == 1

predict p1 
sum p1 if istestdata == "both"
brier default p1 if istestdata == "both"
return list


// With further interactions 
g f_l =string(fico_bin)+"_"+string(ltv_bin)
replace f_l = "800_100" if f_l=="820_100" // (small n)
encode f_l, gen(fico_ltv_bin)
groups fico_ltv_bin  if sample == 1, select(10) order(l)
qui tab fico_ltv_bin, gen(fico_ltv_bin_dum)

tab loan_purpose_dum_*, m
g f_l_p = f_l + "_" + string(loan_purpose_dum_1)
encode f_l_p, gen(fico_ltv_purch_bin)
qui tab fico_ltv_purch_bin, gen(fico_ltv_purch_bin_dum)

egen term15 = rowmax(term_nmon_dum_120 term_nmon_dum_180 )
g f_l_t = f_l + "_" + string(term15)
encode f_l_t, gen(fico_ltv_term_bin)
qui tab fico_ltv_term_bin, gen(fico_ltv_term_bin_dum)

g f_l_d = f_l + "_" + string(document_type_dum_1)
encode f_l_d, gen(fico_ltv_doc_bin)
qui tab fico_ltv_doc_bin, gen(fico_ltv_doc_bin_dum)

logit default orig_amt log_orig_amt occupancy_type_dum_1 occupancy_type_dum_2  jumbo_flg_dum_n investor_type_dum_1 investor_type_dum_2 investor_type_dum_3 ///
 loan_purpose_dum_2 coapplicant_dum_0  term_nmon_dum_180 term_nmon_dum_240 loan_type_mcdash_dum_c orig_year_dum_* prop_state_dum* loan_purpose_dum_1 term_nmon_dum_120 document_type_dum_1 ///
 document_type_dum_2 document_type_dum_3 income_bin_dum* fico_ltv_purch_bin_dum* fico_ltv_term_bin_dum* fico_ltv_doc_bin_dum* ltv_80  if sample == 1 
 
predict p2c 
sum p2c if istestdata == "both"
brier default p2c if istestdata == "both"
return list

pwcorr default p1 p2c if istestdata=="both"

outsheet n default p1 p2c using `output_path'/PD_p1_p2c.csv, comma replace

