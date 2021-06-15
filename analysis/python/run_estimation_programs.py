

import pickle
#from sklearn.externals import joblib
import joblib
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

np.random.seed(seed = 2131985)

from estimation_programs import subset_features
from estimation_programs import load_data
from estimation_programs import read_clf
from estimation_programs import read_clf2
from estimation_programs import estimate_classifier_set
from estimation_programs import estimate_race_classifier_set
from estimation_programs import plot_roc
from estimation_programs import plot_precision
from estimation_programs import predict_defaults
from estimation_programs import plot_calibration
from estimation_programs import cdf_pd_diff
from estimation_programs import bootstrap_models
from estimation_programs import bootstrap_models_resample
from estimation_programs import scramble_datasets

PATH = "../../data/"
output_PATH = "../../output/"

# models
clfs = ["LogitNonLinear","RandomForestIsotonic"]
plotrace = ["Asian","White Non-Hispanic","White Hispanic","Black"]

feature_names_race = {}
feature_names_norace = {}
for model in clfs:
    with open(PATH + "feature_names_race" + model + ".csv") as f:
        reader = csv.reader(f)
        feature_names_race[model] = next(reader)
        feature_names_race[model] = [x.lower() for x in feature_names_race[model]]        
    with open(PATH + "feature_names_norace" + model + ".csv") as f:
        reader = csv.reader(f)
        feature_names_norace[model] = next(reader)
        feature_names_norace[model] = [x.lower() for x in feature_names_norace[model]]
        
# for model in clfs:
#     if model == "LogitNonLinear":
#         with open(PATH + "feature_names_norace" + model + ".csv",'w') as f:
#             writer = csv.writer(f)
#             writer.writerow(list(loaded_data_norace["x_train_nl"].columns))
#         with open(PATH + "feature_names_race" + model + ".csv",'w') as f:
#             writer = csv.writer(f)
#             writer.writerow(list(loaded_data_race["x_train_nl"].columns))
#     if model == "RandomForestIsotonic":
#         with open(PATH + "feature_names_norace" + model + ".csv",'w') as f:
#             writer = csv.writer(f)
#             writer.writerow(list(loaded_data_norace["x_train"].columns))
#         with open(PATH + "feature_names_race" + model + ".csv",'w') as f:
#             writer = csv.writer(f)
#             writer.writerow(list(loaded_data_race["x_train"].columns))




###Output should include:
## 1. Four estimated models [estimation_output_norace["models"]]
## 2. evaluation stats for models (in file and in dictionary) [estimation_output_norace["stats"]
## 3. input for curve graphs [estimation_output_norace["roc_curve"]
##                            and estimation_output_norace["precision_curve"]
## 4. Predicted default rates for full sample in file and in dict [estimated_defaults_norace]
## 5. Four estimated models of RACE [estimatoin_output_raceoutcome]
## 6. evaluation stats for models for race (in file and in dictionary) [estimation_output_raceoutcome["stats"]
## 7. input for curve graphs [estimation_output_raceoutcome["roc_curve"]
##                            and estimation_output_raceoutcome["precision_curve"]
## 8. Write Table comparing models to FILE 
## 9. Write Table comparing race models to FILE 
loaded_data_norace = load_data(PATH, feature_names_norace, race = 0)

## This takes a LONG time to run
#bootstrap_models_resample(loaded_data_norace, feature_names_norace, k=100, int_rate=0)

### MAIN MODEL (with interest rate)
loaded_data_norace = load_data(PATH, feature_names_norace, race = 0, model = 0, int_rate=1)

estimation_output_norace = estimate_classifier_set(output_PATH, loaded_data_norace, race=0, int_rate=1,
                                                   fn_head = "", additional_model = False)

estimated_defaults_norace = predict_defaults(output_PATH,
                                             loaded_data_norace,
                                             estimation_output_norace["models"],
                                             race=0, int_rate=1, fn_head = "",
                                             additional_models = False)


graph_labels = {"Logit" : "Logit", "LogitNonLinear": "Nonlinear Logit", "LogitNonLinear2": "Nonlinear Logit (Full)", "RandomForestIsotonic": "Random Forest"}
graph_labels2 = {"Logit": "Logit", "LogitNonLinear": "Nonlinear Logit", "LogitNonLinear2": "Nonlinear Logit (Full)", "RandomForestIsotonic": "Random Forest (Isotonic)", "RandomForest": "Random Forest"}
plot_roc(output_PATH, estimation_output_norace["roc_curve"], graph_labels, race = 0, int_rate = 1, fn_head = "")
plot_precision(output_PATH, estimation_output_norace["precision_curve"], graph_labels, race = 0, int_rate = 1, fn_head = "")
plot_calibration(output_PATH, loaded_data_norace, estimated_defaults_norace["estimated_test_prob"], graph_labels2, race = 0, int_rate = 1, fn_head = "")

loaded_data_race = load_data(PATH, feature_names_race, race = 1)
loaded_data_norace["y_black_train"] = loaded_data_race["y_black_train"]
loaded_data_norace["y_black_test"] = loaded_data_race["y_black_test"]
loaded_data_norace["y_black_cal"] = loaded_data_race["y_black_cal"] 
estimation_output_raceoutcome = estimate_race_classifier_set(output_PATH,
                                                             loaded_data_norace,
                                                             race=0, int_rate=1,
                                                             fn_head = "")
plot_roc(output_PATH, estimation_output_raceoutcome["roc_curve"], graph_labels, race = 0, int_rate = 1, fn_head = "raceoutcome")
plot_precision(output_PATH, estimation_output_raceoutcome["precision_curve"], graph_labels, race = 0, int_rate = 1, fn_head = "raceoutcome")




### MAIN MODEL (no interest rate)
loaded_data_norace = load_data(PATH, feature_names_norace, race = 0)
estimation_output_norace = estimate_classifier_set(output_PATH, loaded_data_norace, race=0, int_rate=0,
                                                   fn_head = "", additional_model = False)

models = read_clf2(output_PATH, ["Logit", "LogitNonLinear", "RandomForestIsotonic"], 0,0)

## Original bootstrap approach
roc_output, precision_output, brier_output = bootstrap_models(output_PATH, loaded_data_norace, models, k=5, additional_model = False)
estimated_defaults_norace = predict_defaults(output_PATH,
                                             loaded_data_norace,
                                             estimation_output_norace["models"],
                                             race=0, int_rate=0, fn_head = "",
                                             additional_models = False)




graph_labels = {"Logit" : "Logit", "LogitNonLinear": "Nonlinear Logit", "LogitNonLinear2": "Nonlinear Logit (Full)", "RandomForestIsotonic": "Random Forest"}
graph_labels2 = {"Logit": "Logit", "LogitNonLinear": "Nonlinear Logit", "LogitNonLinear2": "Nonlinear Logit (Full)", "RandomForestIsotonic": "Random Forest (Isotonic)", "RandomForest": "Random Forest"}
plot_roc(output_PATH, estimation_output_norace["roc_curve"], graph_labels, race = 0, int_rate = 0, fn_head = "")
plot_precision(output_PATH, estimation_output_norace["precision_curve"], graph_labels, race = 0, int_rate = 0, fn_head = "")
plot_calibration(output_PATH, loaded_data_norace, estimated_defaults_norace["estimated_test_prob"], graph_labels2, race = 0, int_rate = 0, fn_head = "")

## New Bootstrap Approach
# bootstrap_models_resample(loaded_data_norace, feature_names_norace, k=2, int_rate=0)


## Race as outocme
loaded_data_race = load_data(PATH, feature_names_race, race = 1)
loaded_data_norace["y_black_train"] = loaded_data_race["y_black_train"]
loaded_data_norace["y_black_test"] = loaded_data_race["y_black_test"]
loaded_data_norace["y_black_cal"] = loaded_data_race["y_black_cal"] 
estimation_output_raceoutcome = estimate_race_classifier_set(output_PATH,
                                                             loaded_data_norace,
                                                             race=0, int_rate=0,
                                                             fn_head = "")
plot_roc(output_PATH, estimation_output_raceoutcome["roc_curve"], graph_labels, race = 0, int_rate = 0, fn_head = "raceoutcome")
plot_precision(output_PATH, estimation_output_raceoutcome["precision_curve"], graph_labels, race = 0, int_rate = 0, fn_head = "raceoutcome")


### With race
estimation_output_race = estimate_classifier_set(output_PATH, loaded_data_race, race=1, int_rate=0,
                                                 fn_head = "", additional_model = False)
estimated_defaults_race = predict_defaults(output_PATH, loaded_data_race,
                                             estimation_output_race["models"],
                                             race=1, int_rate=0, fn_head = "",
                                             additional_models = False)

plot_roc(output_PATH, estimation_output_race["roc_curve"], graph_labels, race = 1, int_rate = 0, fn_head = "")
plot_precision(output_PATH, estimation_output_race["precision_curve"], graph_labels, race = 1, int_rate = 0, fn_head = "")
plot_calibration(output_PATH, loaded_data_race, estimated_defaults_race["estimated_test_prob"], graph_labels2, race = 1, int_rate = 0, fn_head = "")

#full_data_race = loaded_data_race["full_data"].copy()
full_data_race = load_data(PATH, feature_names_race, race = 1)["full_data"]

racecols = [col for col in list(full_data_race) if col.startswith('race_dum')]
full_data_race["White Non-Hispanic"] = 1 -  full_data_race[racecols].sum(axis=1)
full_data_race["Race"] = full_data_race[racecols + ["White Non-Hispanic"]].idxmax(axis=1).str.replace('race_dum_','').replace('White hisp','White Hispanic')


estimated_defaults_norace = pd.read_csv(output_PATH + "_race0_interestrate0.csv")
for clf in clfs:
    full_data_race[clf] = estimated_defaults_norace[clf]

model_name = ""        
cdf_pd_diff(full_data_race, full_data_race["Race"], plotrace, clfs, log=False)
plt.savefig(output_PATH + "fig_cdf_pd_difference_%s.pdf" % model_name, bbox_inches='tight')
plt.close()
cdf_pd_diff(full_data_race, full_data_race["Race"], plotrace, clfs, log=True)
plt.savefig(output_PATH + "fig_cdf_logpd_difference_%s.pdf" % model_name, bbox_inches='tight')
plt.close()

full_data_all_var = copy.deepcopy(loaded_data_norace)
full_data = copy.deepcopy(full_data_all_var)

### ADDITIONAL MODELS

#1. Drop observations with missing/unknown race

model_name = "model_dropunknown"
loaded_data_norace = load_data(PATH, feature_names_norace, race = 0, model = 1)
    
estimation_output_norace = estimate_classifier_set(output_PATH, loaded_data_norace, race=0, int_rate=0,
                                                   fn_head = model_name)
estimated_defaults_norace = predict_defaults(output_PATH, full_data,
                                             estimation_output_norace["models"],
                                             race=0, int_rate=0, fn_head = model_name)

plot_roc(output_PATH, estimation_output_norace["roc_curve"], graph_labels, race = 0, int_rate = 0, fn_head = model_name)
plot_precision(output_PATH, estimation_output_norace["precision_curve"], graph_labels, race = 0, int_rate = 0, fn_head = model_name)
plot_calibration(output_PATH, full_data, estimated_defaults_norace["estimated_test_prob"], graph_labels2, race = 0, int_rate = 0, fn_head = model_name)

estimated_defaults_norace = pd.read_csv(output_PATH +  model_name + "_race0_interestrate0.csv")
for clf in clfs:
    full_data_race[clf] = estimated_defaults_norace[clf]

cdf_pd_diff(full_data_race, full_data_race["Race"], plotrace, clfs, log=False)
plt.savefig(output_PATH + "fig_cdf_pd_difference_%s.pdf" % model_name, bbox_inches='tight')
plt.close()
cdf_pd_diff(full_data_race, full_data_race["Race"], plotrace, clfs, log=True)
plt.savefig(output_PATH + "fig_cdf_logpd_difference_%s.pdf" % model_name, bbox_inches='tight')
plt.close()



#2. Use 2009-2011 Only
full_data = copy.deepcopy(full_data_all_var)

model_name = "no_fintech1"
loaded_data_norace = load_data(PATH, feature_names_norace, race = 0, model = 2)
loaded_data_features = {}
for x in loaded_data_norace.keys():
    try:
        loaded_data_features[x] = loaded_data_norace[x].columns
    except AttributeError:
        pass

estimation_output_norace = estimate_classifier_set(output_PATH, loaded_data_norace, race=0, int_rate=0,
                                                   fn_head = model_name)
estimated_defaults_norace = predict_defaults(output_PATH, subset_features(full_data, loaded_data_features),
                                             estimation_output_norace["models"],
                                             race=0, int_rate=0, fn_head = model_name)

plot_roc(output_PATH, estimation_output_norace["roc_curve"], graph_labels, race = 0, int_rate = 0, fn_head = model_name)
plot_precision(output_PATH, estimation_output_norace["precision_curve"], graph_labels, race = 0, int_rate = 0, fn_head = model_name)
plot_calibration(output_PATH, full_data, estimated_defaults_norace["estimated_test_prob"], graph_labels2, race = 0, int_rate = 0, fn_head = model_name)

estimated_defaults_norace = pd.read_csv(output_PATH + "" + model_name + "_race0_interestrate0.csv")
for clf in clfs:
    full_data_race[clf] = estimated_defaults_norace[clf]

cdf_pd_diff(full_data_race, full_data_race["Race"], plotrace, clfs, log=False)
plt.savefig(output_PATH + "fig_cdf_pd_difference_%s.pdf" % model_name, bbox_inches='tight')
plt.close()
cdf_pd_diff(full_data_race, full_data_race["Race"], plotrace, clfs, log=True)
plt.savefig(output_PATH + "fig_cdf_logpd_difference_%s.pdf" % model_name, bbox_inches='tight')
plt.close()



## 3. Use Purchase Loans Only
full_data = copy.deepcopy(full_data_all_var)

model_name = "no_fintech2"
loaded_data_norace = load_data(PATH, feature_names_norace, race = 0, model = 3)
loaded_data_features = {}
for x in loaded_data_norace.keys():
    try:
        loaded_data_features[x] = loaded_data_norace[x].columns
    except AttributeError:
        pass
## fix for simulated code    
loaded_data_features["x_train_nl"] = loaded_data_features["x_test_nl"]
#loaded_data_features["full_data"] = loaded_data_features["x_test_nl"]

    
estimation_output_norace = estimate_classifier_set(output_PATH, subset_features(loaded_data_norace, loaded_data_features), race=0, int_rate=0,
                                                   fn_head = model_name)

full_data["dropList_nl"] = list(set(full_data["dropList_nl"]+['income_bin_dum_500']))
estimated_defaults_norace = predict_defaults(output_PATH, subset_features(full_data, loaded_data_features),
                                             estimation_output_norace["models"],
                                             race=0, int_rate=0, fn_head = model_name)

plot_roc(output_PATH, estimation_output_norace["roc_curve"], graph_labels, race = 0, int_rate = 0, fn_head = model_name)
plot_precision(output_PATH, estimation_output_norace["precision_curve"], graph_labels, race = 0, int_rate = 0, fn_head = model_name)
plot_calibration(output_PATH, full_data, estimated_defaults_norace["estimated_test_prob"], graph_labels2, race = 0, int_rate = 0, fn_head = model_name)

estimated_defaults_norace = pd.read_csv(output_PATH + "" + model_name + "_race0_interestrate0.csv")
for clf in clfs:
    full_data_race[clf] = estimated_defaults_norace[clf]

cdf_pd_diff(full_data_race, full_data_race["Race"], plotrace, clfs, log=False)
plt.savefig(output_PATH + "fig_cdf_pd_difference_%s.pdf" % model_name, bbox_inches='tight')
plt.close()
cdf_pd_diff(full_data_race, full_data_race["Race"], plotrace, clfs, log=True)
plt.savefig(output_PATH + "fig_cdf_logpd_difference_%s.pdf" % model_name, bbox_inches='tight')
plt.close()



#4. Use Whites Only
full_data = copy.deepcopy(full_data_all_var)
model_name = "white_only"
loaded_data_norace = load_data(PATH, feature_names_norace, race = 0, model = 4)
loaded_data_features = {}
for x in loaded_data_norace.keys():
    try:
        loaded_data_features[x] = loaded_data_norace[x].columns
    except AttributeError:
        pass

estimation_output_norace = estimate_classifier_set(output_PATH, loaded_data_norace, race=0, int_rate=0,
                                                   fn_head = model_name)
estimated_defaults_norace = predict_defaults(output_PATH, subset_features(full_data, loaded_data_features),
                                             estimation_output_norace["models"],
                                             race=0, int_rate=0, fn_head = model_name)

plot_roc(output_PATH, estimation_output_norace["roc_curve"], graph_labels, race = 0, int_rate = 0, fn_head = model_name)
plot_precision(output_PATH, estimation_output_norace["precision_curve"], graph_labels, race = 0, int_rate = 0, fn_head = model_name)
plot_calibration(output_PATH, full_data, estimated_defaults_norace["estimated_test_prob"], graph_labels2, race = 0, int_rate = 0, fn_head = model_name)

estimated_defaults_norace = pd.read_csv(output_PATH + "" + model_name + "_race0_interestrate0.csv")
for clf in clfs:
    full_data_race[clf] = estimated_defaults_norace[clf]

cdf_pd_diff(full_data_race, full_data_race["Race"], plotrace, clfs, log=False)
plt.savefig(output_PATH + "fig_cdf_pd_difference_%s.pdf" % model_name, bbox_inches='tight')
plt.close()
cdf_pd_diff(full_data_race, full_data_race["Race"], plotrace, clfs, log=True)
plt.savefig(output_PATH + "fig_cdf_logpd_difference_%s.pdf" % model_name, bbox_inches='tight')
plt.close()




#5. GSE full doc only
full_data = copy.deepcopy(full_data_all_var)
model_name = "gse_full"
loaded_data_norace = load_data(PATH, feature_names_norace, race = 0, model = 5)
loaded_data_features = {}
for x in loaded_data_norace.keys():
    try:
        loaded_data_features[x] = loaded_data_norace[x].columns
    except AttributeError:
        pass

### small fix for NL logit due to simulated data
loaded_data_features["x_train_nl"] =  loaded_data_features["x_train_nl"].drop("income_bin_dum_500")
loaded_data_features["x_test_nl"] =  loaded_data_features["x_test_nl"].drop("income_bin_dum_500")
loaded_data_norace["x_train_nl"] = loaded_data_norace["x_train_nl"].drop("income_bin_dum_500", axis = 1)
loaded_data_norace["x_test_nl"] = loaded_data_norace["x_test_nl"].drop("income_bin_dum_500", axis = 1)
full_data["dropList_nl"] = full_data["dropList_nl"] + ["income_bin_dum_500"]

estimation_output_norace = estimate_classifier_set(output_PATH, loaded_data_norace, race=0, int_rate=0,
                                                   fn_head = model_name)
estimated_defaults_norace = predict_defaults(output_PATH, subset_features(full_data, loaded_data_features),
                                             estimation_output_norace["models"],
                                             race=0, int_rate=0, fn_head = model_name)

plot_roc(output_PATH, estimation_output_norace["roc_curve"], graph_labels, race = 0, int_rate = 0, fn_head = model_name)
plot_precision(output_PATH, estimation_output_norace["precision_curve"], graph_labels, race = 0, int_rate = 0, fn_head = model_name)
plot_calibration(output_PATH, full_data, estimated_defaults_norace["estimated_test_prob"], graph_labels2, race = 0, int_rate = 0, fn_head = model_name)

estimated_defaults_norace = pd.read_csv(output_PATH + "" + model_name + "_race0_interestrate0.csv")
for clf in clfs:
    full_data_race[clf] = estimated_defaults_norace[clf]

cdf_pd_diff(full_data_race, full_data_race["Race"], plotrace, clfs, log=False)
plt.savefig(output_PATH + "fig_cdf_pd_difference_%s.pdf" % model_name, bbox_inches='tight')
plt.close()
cdf_pd_diff(full_data_race, full_data_race["Race"], plotrace, clfs, log=True)
plt.savefig(output_PATH + "fig_cdf_logpd_difference_%s.pdf" % model_name, bbox_inches='tight')
plt.close()


#6. No Fico Score
full_data = copy.deepcopy(full_data_all_var)
model_name = "no_fico"

## FIX FOR SIMULATED DATA
full_data["dropList_nl"].remove('fico_orig_fill')
full_data["dropList_nl"].remove('fico_orig_miss')
full_data["dropList_nl"].remove('fico_bin_dum_820')
# for x in ['fico_bin_dum_780', 'fico_bin_dum_600', 'fico_bin_dum_820', 'fico_bin_dum_660', 'fico_bin_dum_720', 'fico_bin_dum_0', 'fico_bin_dum_740', 'fico_bin_dum_640', 'fico_bin_dum_700', 'fico_bin_dum_760', 'fico_bin_dum_680', 'fico_bin_dum_800', 'fico_bin_dum_620']:
#     full_data["dropList_nl"].remove(x)
loaded_data_norace = load_data(PATH, feature_names_norace, race = 0, model = 6)
loaded_data_features = {}
for x in loaded_data_norace.keys():
    if x != "x_train_nl2" and x != "x_test_nl2": 
        try:
            loaded_data_features[x] = loaded_data_norace[x].columns
        except AttributeError:
            pass

estimation_output_norace = estimate_classifier_set(output_PATH, loaded_data_norace, race=0, int_rate=0,
                                                   fn_head = model_name)
estimated_defaults_norace = predict_defaults(output_PATH, subset_features(full_data, loaded_data_features),
                                             estimation_output_norace["models"],
                                             race=0, int_rate=0, fn_head = model_name)

plot_roc(output_PATH, estimation_output_norace["roc_curve"], graph_labels, race = 0, int_rate = 0, fn_head = model_name)
plot_precision(output_PATH, estimation_output_norace["precision_curve"], graph_labels, race = 0, int_rate = 0, fn_head = model_name)
plot_calibration(output_PATH, full_data, estimated_defaults_norace["estimated_test_prob"], graph_labels2, race = 0, int_rate = 0, fn_head = model_name)

estimated_defaults_norace = pd.read_csv(output_PATH + "" + model_name + "_race0_interestrate0.csv")
for clf in clfs:
    full_data_race[clf] = estimated_defaults_norace[clf]

cdf_pd_diff(full_data_race, full_data_race["Race"], plotrace, clfs, log=False)
plt.savefig(output_PATH + "fig_cdf_pd_difference_%s.pdf" % model_name, bbox_inches='tight')
plt.close()
cdf_pd_diff(full_data_race, full_data_race["Race"], plotrace, clfs, log=True)
plt.savefig(output_PATH + "fig_cdf_logpd_difference_%s.pdf" % model_name, bbox_inches='tight')
plt.close()

## XGBoost Graph
# estimated_defaults_xgboost = pd.read_csv(output_PATH + "xgboost_output_w_race_noint.csv")
# estimated_defaults_norace = pd.read_csv(output_PATH + "predictions_race0_interestrate0.csv")

# racecols = [col for col in list(estimated_defaults_xgboost) if col.startswith('race_dum')]
# estimated_defaults_xgboost["White Non-Hispanic"] = 1 -  estimated_defaults_xgboost[racecols].sum(axis=1)
# estimated_defaults_xgboost["Race"] = estimated_defaults_xgboost[racecols + ["White Non-Hispanic"]].idxmax(axis=1).str.replace('race_dum_','').replace('White hisp','White Hispanic')

# clfs = ["LogitNonLinear","xgboost_output"]
# xgboost_comp = estimated_defaults_xgboost["xgboost_output"].to_frame().join(estimated_defaults_norace["LogitNonLinear"].to_frame())
# cdf_pd_diff(xgboost_comp, estimated_defaults_xgboost["Race"], plotrace, clfs, log=False)
# plt.savefig(output_PATH + "fig_cdf_pd_difference_xgboost.pdf", bbox_inches='tight')
# plt.close()
# cdf_pd_diff(xgboost_comp, estimated_defaults_xgboost["Race"], plotrace, clfs, log=True)
# plt.savefig(output_PATH + "fig_cdf_logpd_difference_xgboost.pdf", bbox_inches='tight')
# plt.close()
