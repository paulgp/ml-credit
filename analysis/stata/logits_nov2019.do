
// this is based on sanity_check_pgp.do

cd ~/Dropbox/ML-credit/sato_06112018

set matsize 1000

insheet using all_vals_race1_interestrate1.csv, comma clear names

gen sample = istestdata != "both" & iscalibratedata != "both"
keep if sample == 1

logit default applicant_income income_miss  ltv_ratio_fill fico_orig_miss fico_orig_fill cur_int_rate orig_amt log_orig_amt document_type_dum_1 document_type_dum_2 document_type_dum_3  occupancy_type_dum_1 occupancy_type_dum_2  jumbo_flg_dum_n   investor_type_dum_1 investor_type_dum_2 investor_type_dum_3  loan_purpose_dum_1 loan_purpose_dum_2 coapplicant_dum_0 term_nmon_dum_120 term_nmon_dum_180 term_nmon_dum_240 loan_type_mcdash_dum_c orig_year_dum_* prop_state_dum*



*logit default applicant_income income_miss ltv_ratio_miss ltv_ratio_fill fico_orig_miss fico_orig_fill cur_int_rate orig_amt log_orig_amt document_type_dum_1 document_type_dum_2 document_type_dum_3  occupancy_type_dum_1 occupancy_type_dum_2  jumbo_flg_dum_n   investor_type_dum_0 investor_type_dum_1 investor_type_dum_2 investor_type_dum_3 investor_type_dum_4 investor_type_dum_6   loan_purpose_dum_1 loan_purpose_dum_2 loan_purpose_dum_3 coapplicant_dum_0 coapplicant_dum_1 term_nmon_dum_120 term_nmon_dum_180 term_nmon_dum_240 term_nmon_dum_360 loan_type_mcdash_dum_c loan_type_mcdash_dum_d orig_year_dum_2011 msa_id_dum_*

*esttab using stata_coef.csv, replace csv
/*
// estimate on full sample (i.e. don't leave out test sample)
rename ever90dpd default

g log_orig_amt = ln(orig_amt)

foreach x in document_type occupancy_type jumbo_flg investor_type loan_purpose coapplicant ///
 term_nmon fico_bin ltv_bin ltv_80 income_bin loan_type_mcdash msa_id {
qui tab `x', gen(`x'_) 
}

logit default cur_int_rate orig_amt log_orig_amt document_type_* occupancy_type_* jumbo_flg_* investor_type_* loan_purpose_* coapplicant_* term_nmon_* ///
 fico_bin_* ltv_bin_* ltv_80 income_bin_* loan_type_mcdash_* // msa_id_*    
//if test  != "both"



logit default cur_int_rate orig_amt log_orig_amt ///
fico_orig ltv_ratio applicant_income  //  loan_type_mcdash_*msa_id_*   document_type_* occupancy_type_* jumbo_flg_* investor_type_* loan_purpose_* coapplicant_* term_nmon_* /// 
// this converges very quickly
estat classification
estat gof
lroc , nograph
lsens, nograph


logit default cur_int_rate orig_amt log_orig_amt ///
fico_bin_* ltv_ratio applicant_income  //  loan_type_mcdash_*msa_id_*   document_type_* occupancy_type_* jumbo_flg_* investor_type_* loan_purpose_* coapplicant_* term_nmon_* /// 
// JUST ADDING FICO BIN CREATES PROBLEMS

replace fico_bin =600 if inrange(fico_bin,1,599)
replace fico_bin =820 if fico_bin==840
tab fico_bin, gen(fico_bin_)
logit default cur_int_rate orig_amt log_orig_amt ///
fico_bin_* ltv_ratio applicant_income  //  loan_type_mcdash_*msa_id_*   document_type_* occupancy_type_* jumbo_flg_* investor_type_* loan_purpose_* coapplicant_* term_nmon_* /// 
// THIS WORKS NO PROBLEM

logit default cur_int_rate orig_amt log_orig_amt ///
fico_bin_* ltv_bin_* ltv_80 applicant_income
// STILL NO PROBLEM

logit default cur_int_rate orig_amt log_orig_amt ///
fico_bin_* ltv_bin_* ltv_80 income_bin_*
// STILL FINE

logit default cur_int_rate orig_amt log_orig_amt ///
fico_bin_* ltv_bin_* ltv_80 income_bin_* ///
loan_type_mcdash_*  document_type_* occupancy_type_* jumbo_flg_* investor_type_* loan_purpose_* coapplicant_* term_nmon_* 
// STILL FINE


logit default cur_int_rate orig_amt log_orig_amt document_type_* occupancy_type_* jumbo_flg_* investor_type_* loan_purpose_* coapplicant_* term_nmon_* ///
 fico_bin_* ltv_bin_* ltv_80 income_bin_* loan_type_mcdash_* msa_id_*    
// converges fine; takes about 25 minutes; AUC=0.8522


// now redo with estimation sample and test sample (same size as Paul's)
set seed 98034
generate u1 = runiform()

logit default cur_int_rate orig_amt log_orig_amt document_type_* occupancy_type_* jumbo_flg_* investor_type_* loan_purpose_* coapplicant_* term_nmon_* ///
 fico_bin_* ltv_bin_* ltv_80 income_bin_* loan_type_mcdash_* msa_id_* if u1<=0.49
predict lnl_race0

lroc if u1<=0.49, nograph // estimation sample -- 0.8493
lroc if u1> 0.7,  nograph // test sample -- 0.8509
estat classification if u1> 0.7

// add race:
logit default cur_int_rate orig_amt log_orig_amt document_type_* occupancy_type_* jumbo_flg_* investor_type_* loan_purpose_* coapplicant_* term_nmon_* ///
 fico_bin_* ltv_bin_* ltv_80 income_bin_* loan_type_mcdash_* msa_id_* i.race_n if u1<=0.49
predict lnl_race1

lroc if u1<=0.49, nograph // estimation sample -- 
lroc if u1> 0.7,  nograph // test sample -- 0.8515

tabstat lnl_race0 lnl_race1 if u1> 0.7, by(race_n) format(%9.3g)

////////////////////
// simpler version
replace fico_orig = 0 if 0.fico_bin==1 // missings
logit default cur_int_rate orig_amt  ///
 fico_orig 0.fico_bin ltv_ratio  if u1<=0.49

lroc if u1<=0.49, nograph // estimation sample
lroc if u1> 0.7,  nograph // test sample -- 0.8161

//add race:
logit default cur_int_rate orig_amt  ///
 fico_orig 0.fico_bin ltv_ratio i.race_n if u1<=0.49

lroc if u1<=0.49, nograph // estimation sample
lroc if u1> 0.7,  nograph // test sample



// estimate on purchase only:
logit default cur_int_rate orig_amt log_orig_amt document_type_* occupancy_type_* jumbo_flg_* investor_type_*  coapplicant_* term_nmon_* ///
 fico_bin_* ltv_bin_* ltv_80 income_bin_* loan_type_mcdash_* msa_id_* if u1<=0.49 & loan_purpose==1

lroc if u1<=0.49 & loan_purpose==1, nograph // estimation sample -- 
lroc if u1> 0.7  & loan_purpose==1, nograph // test sample -- 




/////////////////////
// Different origination year:
use data/sample2009to2014_allinvtypes_clean.dta, clear

// estimate on full sample (i.e. don't leave out test sample)
rename ever90dpd default

g log_orig_amt = ln(orig_amt)

foreach x in document_type occupancy_type jumbo_flg investor_type loan_purpose coapplicant ///
 term_nmon ltv_bin ltv_80 income_bin loan_type_mcdash msa_id {
qui tab `x', gen(`x'_) 
}
replace fico_bin =600 if inrange(fico_bin,1,599)
replace fico_bin =820 if fico_bin==840
tab fico_bin, gen(fico_bin_)

set seed 98034
generate u1 = runiform()

keep if action_qtr>=d(1jan2012) & action_qtr<=d(31dec2012)


logit default cur_int_rate orig_amt log_orig_amt document_type_* occupancy_type_* jumbo_flg_* investor_type_* loan_purpose_* coapplicant_* term_nmon_* ///
 fico_bin_* ltv_bin_* ltv_80 income_bin_* loan_type_mcdash_* msa_id_* if u1<=0.49
predict lnl_race0

lroc if u1<=0.49, nograph // estimation sample -- 
lroc if u1> 0.7,  nograph // test sample -- 
estat classification if u1> 0.7

// add race:
logit default cur_int_rate orig_amt log_orig_amt document_type_* occupancy_type_* jumbo_flg_* investor_type_* loan_purpose_* coapplicant_* term_nmon_* ///
 fico_bin_* ltv_bin_* ltv_80 income_bin_* loan_type_mcdash_* msa_id_* i.race_n if u1<=0.49
predict lnl_race1

lroc if u1<=0.49, nograph // estimation sample -- 
lroc if u1> 0.7,  nograph // test sample -- 

tabstat lnl_race0 lnl_race1 if u1> 0.7, by(race_n) format(%9.3g)






/*

predict logitnonlinear_stata if test == "both"




gen log_logitnonlinear = log(logitnonlinear)
gen log_rf = log(randomforestisotonic)

gen diff_pd = log_rf - log_logitnonlinear

gen diff_pd2 = randomforestisotonic - logitnonlinear


binscatter log_logitnonlinaer log_rf cur_int_rate
binscatter logitnonlinear randomforestisotonic cur_int_rate
binscatter diff_pd cur_int_rate
binscatter diff_pd2 cur_int_rate


