#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:15:52 2018

@author: Paul G-P
"""
import pickle
#from sklearn.externals import joblib
import joblib
import pandas as pd
import statsmodels.api as sm
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
sns.set(style = "whitegrid", palette = sns.color_palette("cubehelix",4))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

seed = 5
def estimate_fit(results_roc, y_test):
    fpr, tpr, _ = roc_curve(y_test, results_roc)
    precision, recall, _ = precision_recall_curve(y_test, results_roc)
    roc_output = metrics.roc_auc_score(y_test, results_roc)
    precision_output = average_precision_score(y_test, results_roc)
    brier_output = brier_score_loss(y_test, results_roc)
    print("Brier score: %f " % (brier_output))
    print("ROC score: %f " % (roc_output))
    print("Average Precision Score: %f " % (precision_output))
    return results_roc, fpr, tpr, precision, recall, roc_output, precision_output, brier_output

def read_clf(path="",names = ["Logit","LogitNonLinear","RandomForestIsotonic"] ,rate_select=1,race_select=0):
    ML_features = list(pd.read_csv(path + "sato_varnames_race%d.csv" % race_select).columns)
    # load all models in specified list, extract features names, store in dict's
    # arguments rate_select and race_select specify what's included
    # default is : interest rate, no race
    models = {}
    features = {}
    for name in names:
        if name == "LogitNonLinear" or name == "Logit":
            with open(path + "%s_race%d_interestrate%d.pkl" 
                      % (name, race_select, rate_select), 'rb') as f:
                clf0 = pickle.load(f)
            # extract feature names
            features[name] = list(clf0.params.index)
        else:
            # set feature names to standard ML list defined above
            features[name] = ML_features
    return models, features

def read_clf2(path="",names = ["Logit","LogitNonLinear","RandomForestIsotonic"] ,rate_select=1,race_select=0):
    # load all models in specified list, extract features names, store in dict's
    # arguments rate_select and race_select specify what's included
    # default is : interest rate, no race
    models = {}
    for name in names:
        if name == "LogitNonLinear" or name == "Logit":
            with open(path + "%s_race%d_interestrate%d.pkl" 
                      % (name, race_select, rate_select), 'rb') as f:
                clf0 = pickle.load(f)
        else:
            with open(path + "%s_race%d_interestrate%d.pkl" 
                      % (name, race_select, rate_select), 'rb') as f:
                clf0 = joblib.load(f)
        models[name] = clf0
    return models

def clean(smpl,verbose = True,add_features=True):
    if add_features:
        ## Adds features to data needed for using nonlinear logit (bins of income, fico, ltv)
        # Not making copy to save time; only use this if happy with a view
        smpl =  sm.add_constant(smpl, prepend=False,  has_constant='add')
        if verbose: print("Making bins")
        ## Replicating STATA code to make bins:
        #egen fico_bin = cut(fico_orig), at(280(20)870)
        #replace fico_bin = 0 if fico_bin == .
        try:
            fico_cuts = [0] + list(range(280,870,20))
            fico_bin = pd.cut(smpl["fico_orig_fill"], fico_cuts, labels = fico_cuts[0:-1], right = False).fillna(0)
            fico_bin.loc[(fico_bin>0) & (fico_bin < 600),] = 600
            fico_bin.loc[fico_bin==840,] = 820
            fico_bin = fico_bin.astype('int')
        except KeyError:
            print("Fico Data Cuts issue")
        #egen ltv_bin = cut(ltv_ratio), at(20(5)105)
        #gen ltv_80 = ltv_ratio == 80
        ltv_cuts = list(range(20,110,5))
        ltv_bin = pd.cut(smpl["ltv_ratio_fill"], ltv_cuts, labels = ltv_cuts[0:-1], right = False)
        ltv_bin = ltv_bin.astype('int')        
        ltv_80 = (smpl["ltv_ratio_fill"]==80).astype(int)    
        #egen income_bin = cut(applicant_income), at(-25(25)525)
        inc_cuts = list(range(-25,550,25))
        income_bin = pd.cut(smpl["applicant_income"],inc_cuts, labels = inc_cuts[0:-1],right = False)
        income_bin = income_bin.astype('int')
        #replace fico_bin =600 if inrange(fico_bin,1,599)
        #replace fico_bin =820 if fico_bin==840
        ## Convert to dummies and join
        if verbose: print("Merging bins with data")
        try:
            smpl = smpl.join(pd.get_dummies(fico_bin, prefix = "fico_bin_dum"))
        except:
            print("Fico Data Cuts join issue ")            
        smpl = smpl.join(pd.get_dummies(income_bin, prefix = "income_bin_dum"))
        smpl = smpl.join(pd.get_dummies(ltv_bin, prefix = "ltv_bin_dum"))
        smpl["ltv_80_dum_0"] = 1 - ltv_80
    
    if verbose: print("Creating year and race variables")
    # undummy year variable
    #years = [col for col in smpl.columns if col.startswith('orig_year_dum_')]
    #smpl["Year"] = smpl[years].idxmax(axis=1).str.replace('orig_year_dum_','').astype('int')
    # undummy race variable
    #races = [col for col in list(smpl) if col.startswith('race_dum')]
    #smpl["White Non-Hispanic"] = 1 -  smpl[races].sum(axis=1)
    #smpl["Race"] = smpl[races + ["White Non-Hispanic"]].idxmax(axis=1).str.replace('race_dum_','').replace('White hisp','White Hispanic')
    return smpl


def interact_bins(smpl2):
    smpl = smpl2.copy(deep=True)
    try:
        a = smpl.columns[smpl.columns.str.startswith('fico_bin_dum')]
        b = smpl.columns[smpl.columns.str.startswith('ltv_bin_dum')]
        
        for col1 in b:
            for col2 in a:
                smpl[col2 + '_' + col1.split('_')[3]] = smpl[col1].mul(smpl[col2])
        #replace f_l = "800_100" if f_l=="820_100"
        smpl["fico_bin_dum_800_100"] = smpl["fico_bin_dum_800_100"] + smpl["ltv_bin_dum_100"]
        smpl = smpl.drop(["ltv_bin_dum_100"], axis=1)
        #smpl = smpl.drop([a,b],axis=1)
    except KeyError:
        print("Interact Bins Failed")
        # for col1 in b:
        #     for col2 in c:
        #         smpl[col2 + '_' + col1.split('_')[3]] = smpl[col1].mul(smpl[col2])

        # for col1 in a:
        #     for col2 in c:
        #         smpl[col2 + '_' + col1.split('_')[3]] = smpl[col1].mul(smpl[col2])
            
    return smpl

def subset_features(full_data, subset_features_dict):
    subset_data = full_data
    for key in subset_features_dict.keys():
        print(key)
        subset_data[key] = full_data[key][subset_features_dict[key]]
    return subset_data
    

def load_data(path="", feature_names=[], race = 1, model = 0, int_rate = 0):
    full_data = pd.read_csv(path + "all_vals_race1_interestrate1.csv")
    if model == 0:
        full_data = full_data
    elif model == 1:
        # No Missing Race
        full_data = full_data.loc[full_data["race_dum_Unknown"] == 0]
    elif model == 2:
        # Filter to 2009-2011
        full_data = full_data.loc[(full_data["orig_year_dum_2009"] == 1) | (full_data["orig_year_dum_2010"] == 1 )| ( full_data["orig_year_dum_2011"] == 1 ) ]
        full_data = full_data.drop(["orig_year_dum_2012", "orig_year_dum_2011"], axis = 1)
    elif model == 3:
        # Purchase Loans Only
        full_data = full_data.loc[(full_data["loan_purpose_dum_1"] == 1)]
        full_data = full_data.drop(["loan_purpose_dum_1", "loan_purpose_dum_2"], axis = 1)
    elif model == 4:
        # Only Estimate on Whites
        full_data = full_data.loc[(full_data["race_dum_Asian"] == 0) & (full_data["race_dum_Unknown"] == 0) & (full_data["race_dum_Black"] == 0) & (full_data["race_dum_Native Am, Alaska, Hawaii"] == 0) & (full_data["race_dum_White hisp"] == 0) ]
    elif model == 5:
        # Only Estimate on GSE FULL DOC
        full_data = full_data.loc[(full_data["document_type_dum_1"] == 1) & ((full_data["investor_type_dum_2"] == 1) | (full_data["investor_type_dum_3"] == 1))]
        full_data = full_data.drop(["document_type_dum_1", "document_type_dum_2", "document_type_dum_3", "investor_type_dum_1", "investor_type_dum_2"], axis = 1)
    elif model == 6:
        # Drop Fico Scores
        full_data = full_data.drop(["fico_orig_fill", "fico_orig_miss"], axis = 1)
    if race == 0 or model == 4:
        full_data = full_data.drop(["race_dum_Unknown", "race_dum_Asian", "race_dum_Black", "race_dum_Native Am, Alaska, Hawaii", "race_dum_White hisp"], axis = 1)
    if model == 1 & race == 1: 
        full_data = full_data.drop(["race_dum_Unknown"], axis = 1)
    try:
        dropList = ['Unnamed: 0', 'Default2', 'RandomForest', 'LogitNonLinear','RandomForestIsotonic', 'cur_int_rate']
        full_data = full_data.drop(dropList,axis=1)
    except KeyError:
        ### From simulated code
        dropList = ['v1', 'istestdata2', 'iscalibratedata2', 'cur_int_rate']
        full_data = full_data.drop(dropList,axis=1)
            

    #'IsTestData', 'Default2',    'IsCalibrateData'

    if int_rate == 1:
        dropList2 = ['IsTestData', 'Default',    'IsCalibrateData']
    else:
        dropList2 = ['IsTestData', 'Default',    'IsCalibrateData','sato']

    y_train = full_data.loc[(full_data['IsTestData'] == 'left_only') & (full_data['IsCalibrateData'] == 'left_only'), 'Default']
    x_train = full_data.loc[(full_data['IsTestData'] == 'left_only') & (full_data['IsCalibrateData'] == 'left_only')].drop(dropList2, axis=1)
    x_train_nl = clean(x_train)
    dropList_nl = list(set(x_train_nl.columns) - set(feature_names["LogitNonLinear"]))
    x_train_nl = x_train_nl.drop(dropList_nl, axis=1)
    x_train_nl2 = interact_bins(x_train_nl)    

    y_cal = full_data.loc[full_data['IsCalibrateData'] == 'both', 'Default']
    x_cal = full_data.loc[full_data['IsCalibrateData'] == 'both'].drop(dropList2, axis=1)
    x_cal_nl = clean(x_cal).drop(dropList_nl, axis=1)

    y_test = full_data.loc[full_data['IsTestData'] == 'both', 'Default']
    x_test = full_data.loc[full_data['IsTestData'] == 'both'].drop(dropList2, axis=1)
    x_test_nl = clean(x_test).drop(dropList_nl, axis=1)
    x_test_nl2 = interact_bins(x_test_nl)    
    
    if race == 1:
        race_data = generate_race_outcome(full_data)
        return {"y_train" : y_train, "y_cal" : y_cal, "y_test" : y_test,
                "x_train" : x_train, "x_cal" : x_cal, "x_test" : x_test,
                "x_train_nl" : x_train_nl,  "x_cal_nl" : x_cal_nl, "x_test_nl" : x_test_nl,
                "x_train_nl2" : x_train_nl2,  "x_test_nl2" : x_test_nl2,                                
                "full_data" : full_data, "dropList" : dropList2, "dropList_nl" : dropList_nl,
                "y_black_train" : race_data["y_black_train"],
                "y_black_cal"   : race_data["y_black_cal"],
                "y_black_test"  : race_data["y_black_test"]}
    else:
        return {"y_train" : y_train, "y_cal" : y_cal, "y_test" : y_test,
                "x_train" : x_train, "x_cal" : x_cal, "x_test" : x_test,
                "x_train_nl" : x_train_nl,  "x_cal_nl" : x_cal_nl, "x_test_nl" : x_test_nl,
                "x_train_nl2" : x_train_nl2,  "x_test_nl2" : x_test_nl2,                
                "full_data" : full_data, "dropList" : dropList2, "dropList_nl" : dropList_nl}

def generate_race_outcome(full_data):
    y_black_train = full_data.loc[(full_data['IsTestData'] == 'left_only') & (full_data['IsCalibrateData'] == 'left_only'), 'race_dum_Black'] + full_data.loc[(full_data['IsTestData'] == 'left_only') & (full_data['IsCalibrateData'] == 'left_only'), 'race_dum_White hisp']
    y_black_cal = full_data.loc[(full_data['IsCalibrateData'] == 'both'), 'race_dum_Black'] + full_data.loc[(full_data['IsCalibrateData'] == 'both'), 'race_dum_White hisp']
    y_black_test = full_data.loc[(full_data['IsTestData'] == 'both'), 'race_dum_Black'] + full_data.loc[(full_data['IsTestData'] == 'both'), 'race_dum_White hisp']    
    return {"y_black_train" : y_black_train,
            "y_black_cal"   : y_black_cal,
            "y_black_test"  : y_black_test}

def predict_defaults(path = "", loaded_data={}, models=0, race=0, int_rate=0, fn_head = "predictions", additional_models = False):
    full_data     = loaded_data["full_data"].drop(loaded_data["dropList"], axis=1)
    full_data_nl  = clean(full_data)
    full_data_nl  = full_data_nl.drop(loaded_data["dropList_nl"], axis=1)
    full_data_nl2 = interact_bins(full_data_nl)
    if additional_models:
        names = [ "Logit",
                  "LogitNonLinear",
                  "LogitNonLinear2",              
                  "RandomForest",
                  "RandomForestIsotonic"
        ]
    else:
        names = [ "Logit",
                  "LogitNonLinear",
                  "RandomForest",
                  "RandomForestIsotonic"
        ]

    predict_data_logit = sm.add_constant(full_data_nl, prepend = False)
    predict_data_logit2 = sm.add_constant(full_data_nl2, prepend = False)    
    estimated_full_prob = {}
    for name in names:
        clf = models[name]
        if name == "LogitNonLinear":
            estimated_full_prob[name] = clf.predict(predict_data_logit)
        elif name == "LogitNonLinear2":
            estimated_full_prob[name] = clf.predict(predict_data_logit2)
        elif name == "Logit":
            estimated_full_prob[name] = clf.predict(sm.add_constant(full_data, prepend = False))
        else:
            estimated_full_prob[name] = clf.predict_proba(full_data)[:,1]

    estimated_test_prob = {}
    for name in names:
        clf = models[name]
        if name == "LogitNonLinear":
            estimated_test_prob[name] = clf.predict(loaded_data["x_test_nl"])
        elif name == "LogitNonLinear2":
            estimated_test_prob[name] = clf.predict(loaded_data["x_test_nl2"])
        elif name == "Logit":
            estimated_test_prob[name] = clf.predict(sm.add_constant(loaded_data["x_test"], prepend = False))            
        else:
            estimated_test_prob[name] = clf.predict_proba(loaded_data["x_test"])[:,1]
            
    pd.DataFrame.from_dict(estimated_full_prob).to_csv(path + fn_head + "_race%d_interestrate%d.csv" % (race, int_rate))
    return {"estimated_full_prob" : estimated_full_prob ,
            "estimated_test_prob" : estimated_test_prob }

def scramble_datasets(full_data, feature_names, int_rate):
    if int_rate == 1:
        dropList2 = ['IsTestData', 'Default',    'IsCalibrateData']
    else:
        dropList2 = ['IsTestData', 'Default',    'IsCalibrateData','sato']

    random_idx = (np.random.random_sample(np.shape(full_data)[0]))
    test_sample = (random_idx  > 0.7)
    train_sample = (random_idx  <= 0.7) * (random_idx > 0.7*0.3)
    calibration_sample = (random_idx <= 0.7*0.3)

    y_train = full_data.loc[train_sample, 'Default']
    x_train = full_data.loc[train_sample].drop(dropList2, axis=1)
    x_train_nl = clean(x_train)
    dropList_nl = list(set(x_train_nl.columns) - set(feature_names["LogitNonLinear"]))
    x_train_nl = x_train_nl.drop(dropList_nl, axis=1)
    x_train_nl2 = interact_bins(x_train_nl)    

    y_cal = full_data.loc[calibration_sample, 'Default']
    x_cal = full_data.loc[calibration_sample].drop(dropList2, axis=1)
    x_cal_nl = clean(x_cal).drop(dropList_nl, axis=1)

    y_test = full_data.loc[test_sample, 'Default']
    x_test = full_data.loc[test_sample].drop(dropList2, axis=1)
    x_test_nl = clean(x_test).drop(dropList_nl, axis=1)
    x_test_nl2 = interact_bins(x_test_nl)    

    return {"y_train" : y_train, "y_cal" : y_cal, "y_test" : y_test,
            "x_train" : x_train, "x_cal" : x_cal, "x_test" : x_test,
            "x_train_nl" : x_train_nl,  "x_cal_nl" : x_cal_nl, "x_test_nl" : x_test_nl,
            "x_train_nl2" : x_train_nl2,  "x_test_nl2" : x_test_nl2,                
            "full_data" : full_data, "dropList" : dropList2, "dropList_nl" : dropList_nl}


def bootstrap_datasets(loaded_data, feature_names, int_rate):
    if int_rate == 1:
        dropList2 = ['IsTestData', 'Default',    'IsCalibrateData']
    else:
        dropList2 = ['IsTestData', 'Default',    'IsCalibrateData','sato']

    random_idx = np.random.choice(np.shape(loaded_data["y_train"])[0], size = np.shape(loaded_data["y_train"])[0])

    if int_rate == 1:
        dropList2 = ['IsTestData', 'Default',    'IsCalibrateData']
    else:
        dropList2 = ['IsTestData', 'Default',    'IsCalibrateData','sato']

    y_train = loaded_data["y_train"].iloc[random_idx]
    x_train = loaded_data["x_train"].iloc[random_idx]
    x_train_nl = loaded_data["x_train_nl"].iloc[random_idx]
    dropList_nl = loaded_data["dropList_nl"]
    x_train_nl = loaded_data["x_train_nl"].iloc[random_idx]
    x_train_nl2 = loaded_data["x_train_nl2"].iloc[random_idx]

    y_cal = loaded_data["y_cal"]
    x_cal = loaded_data["x_cal"]
    x_cal_nl = loaded_data["x_cal_nl"]

    y_test = loaded_data["y_test"]
    x_test = loaded_data["x_test"]
    x_test_nl = loaded_data["x_test_nl"]
    x_test_nl2 = loaded_data["x_test_nl2"]

    return {"y_train" : y_train, "y_cal" : y_cal, "y_test" : y_test,
            "x_train" : x_train, "x_cal" : x_cal, "x_test" : x_test,
            "x_train_nl" : x_train_nl,  "x_cal_nl" : x_cal_nl, "x_test_nl" : x_test_nl,
            "x_train_nl2" : x_train_nl2,  "x_test_nl2" : x_test_nl2,                
            "full_data" : loaded_data["full_data"], "dropList" : dropList2, "dropList_nl" : dropList_nl}


def estimate_classifier_set(path="output/", loaded_data={}, race=0, int_rate=0, fn_head = "", additional_model = False, save_model = True):
        
    clf_forest = RandomForestClassifier(n_estimators=2,
                                        max_depth=None,
                                        min_samples_split=200,
                                        min_samples_leaf = 100,
                                        random_state=seed,
                                        bootstrap = False,
                                        verbose = 1,
                                        n_jobs=-1)

    if additional_model:
        names = ["Logit",
                 "LogitNonLinear",
                 "LogitNonLinear2",             
                 "RandomForest",
                 "RandomForestIsotonic"
        ]
        classifiers = [ sm.Logit(loaded_data["y_train"], sm.add_constant(loaded_data["x_train"], prepend=False)),
                    sm.Logit(loaded_data["y_train"], sm.add_constant(loaded_data["x_train_nl"], prepend=False)),
                    sm.Logit(loaded_data["y_train"], sm.add_constant(loaded_data["x_train_nl2"], prepend=False)),
                    clf_forest,
                    CalibratedClassifierCV(clf_forest,
                                           method='isotonic',
                                           cv = "prefit")]
    else:
        names = ["Logit",
                 "LogitNonLinear",
                 "RandomForest",
                 "RandomForestIsotonic"
        ]
        classifiers = [ sm.Logit(loaded_data["y_train"], sm.add_constant(loaded_data["x_train"], prepend=False)),
                    sm.Logit(loaded_data["y_train"], sm.add_constant(loaded_data["x_train_nl"], prepend=False)),
                    clf_forest,
                    CalibratedClassifierCV(clf_forest,
                                           method='isotonic',
                                           cv = "prefit")]
    
    fpr= dict()
    tpr = dict()
    precision= dict()
    recall = dict()
    roc_output = dict()
    precision_output = dict()
    brier_output = dict()
    results_roc = dict()

    models = dict(zip(names, classifiers))
    for name in names:
        clf = models[name]
        print("Fitting %s" % name)
        if name == "LogitNonLinear":
            res1 = clf.fit()
            print(res1.summary())
            models[name] = res1
            y_hat_test = pd.DataFrame(data=res1.predict(sm.add_constant(loaded_data["x_test_nl"], prepend=False)))
            results_roc[name], fpr[name], tpr[name], precision[name], recall[name], roc_output[name], precision_output[name], brier_output[name] = estimate_fit(y_hat_test, loaded_data["y_test"])
        elif name == "LogitNonLinear2":
            res1 = clf.fit()
            print(res1.summary())
            models[name] = res1
            y_hat_test = pd.DataFrame(data=res1.predict(sm.add_constant(loaded_data["x_test_nl2"], prepend=False)))
            results_roc[name], fpr[name], tpr[name], precision[name], recall[name], roc_output[name], precision_output[name], brier_output[name] = estimate_fit(y_hat_test, loaded_data["y_test"])
        elif name == "Logit":
            res1 = clf.fit()
            print(res1.summary())
            models[name] = res1
            y_hat_test = pd.DataFrame(data=res1.predict(sm.add_constant(loaded_data["x_test"], prepend=False)))            
            results_roc[name], fpr[name], tpr[name], precision[name], recall[name], roc_output[name], precision_output[name], brier_output[name] = estimate_fit(y_hat_test, loaded_data["y_test"])
        else:
            if "Sigmoid" not in name and "Isotonic" not in name:
                print("Base Calibration")
                clf.fit(loaded_data["x_train"], loaded_data["y_train"])
                y_hat_test = pd.DataFrame(data=clf.predict_proba(loaded_data["x_test"]))[1]
                results_roc[name], fpr[name], tpr[name], precision[name], recall[name], roc_output[name], precision_output[name], brier_output[name] = estimate_fit(y_hat_test, loaded_data["y_test"])
            else:
                print("CV Calibration")
                clf.fit(loaded_data["x_cal"], loaded_data["y_cal"])
                y_hat_test = pd.DataFrame(data=clf.predict_proba(loaded_data["x_test"]))[1]
                results_roc[name], fpr[name], tpr[name], precision[name], recall[name], roc_output[name], precision_output[name], brier_output[name] = estimate_fit(y_hat_test, loaded_data["y_test"])
        if save_model:
            if name == "LogitNonLinear2" or name == "LogitNonLinear" or name == "Logit":
                with open(path + fn_head + "%s_race%d_interestrate%d.pkl" % (name, race, int_rate), 'wb') as f:
                    res1.save(f, remove_data=True)
            else:
                with open(path + fn_head + "%s_race%d_interestrate%d.pkl" % (name, race, int_rate), 'wb') as f: 
                    joblib.dump(clf,f)

    with open(path + fn_head + "eval_output_race%d_interestrate%d.csv" % (race, int_rate), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model", "roc", "precision", "brier_score"])
        for name in names:
            writer.writerow([name, roc_output[name], precision_output[name], brier_output[name]])


    return {"models" : models,
            "roc_curve" : {"fpr" : fpr, "tpr" : tpr},
            "precision_curve" : {"precision" : precision, "recall": recall},
            "stats" : [roc_output, precision_output, brier_output] }


def estimate_classifier_set_newversion(loaded_data, race=0, int_rate=0, fn_head = "", additional_model = False):
        
    clf_forest = RandomForestClassifier(n_estimators=2,
                                        max_depth=None,
                                        min_samples_split=200,
                                        min_samples_leaf = 100,
                                        random_state=seed,
                                        bootstrap = False,
                                        verbose = 1,
                                        n_jobs=-1)

    LogitSK = LogisticRegression(penalty="none", solver = "lbfgs").fit(X = loaded_data_norace["x_train"], y = loaded_data_norace["y_train"])
    ## THIS IS NOT DOABLE IN OUR OLD VERSION -- WE WOULD NEED TO UPDATE TO A NEWER VERSION OF SKLEARN
    if additional_model:
        names = ["Logit",
                 "LogitNonLinear",
                 "LogitNonLinear2",             
                 "RandomForest",
                 "RandomForestIsotonic"
        ]
        classifiers = [ sm.Logit(loaded_data["y_train"], sm.add_constant(loaded_data["x_train"], prepend=False)),
                    sm.Logit(loaded_data["y_train"], sm.add_constant(loaded_data["x_train_nl"], prepend=False)),
                    sm.Logit(loaded_data["y_train"], sm.add_constant(loaded_data["x_train_nl2"], prepend=False)),
                    clf_forest,
                    CalibratedClassifierCV(clf_forest,
                                           method='isotonic',
                                           cv = "prefit")]
    else:
        names = ["Logit",
                 "LogitNonLinear",
                 "RandomForest",
                 "RandomForestIsotonic"
        ]
        classifiers = [ sm.Logit(loaded_data["y_train"], sm.add_constant(loaded_data["x_train"], prepend=False)),
                    sm.Logit(loaded_data["y_train"], sm.add_constant(loaded_data["x_train_nl"], prepend=False)),
                    clf_forest,
                    CalibratedClassifierCV(clf_forest,
                                           method='isotonic',
                                           cv = "prefit")]
    
    fpr= dict()
    tpr = dict()
    precision= dict()
    recall = dict()
    roc_output = dict()
    precision_output = dict()
    brier_output = dict()
    results_roc = dict()

    models = dict(zip(names, classifiers))
    for name in names:
        clf = models[name]
        print("Fitting %s" % name)
        if name == "LogitNonLinear":
            res1 = clf.fit()
            print(res1.summary())
            models[name] = res1
            results_roc[name], fpr[name], tpr[name], precision[name], recall[name], roc_output[name], precision_output[name], brier_output[name] = estimate_fit(res1 , sm.add_constant(loaded_data["x_test_nl"], prepend = False), loaded_data["y_test"])
            models[name + "Isotonic"] = res2
            print(res2.summary())
            results_roc["LogitNonLinearIsotonic"], fpr["LogitNonLinearIsotonic"], tpr["LogitNonLinearIsotonic"], precision["LogitNonLinearIsotonic"], recall["LogitNonLinearIsotonic"], roc_output["LogitNonLinearIsotonic"], precision_output["LogitNonLinearIsotonic"], brier_output["LogitNonLinearIsotonic"] = estimate_fit(res2, loaded_data["x_test_nl"], loaded_data["y_test"])                
        elif name == "LogitNonLinear2":
            res1 = clf.fit()
            print(res1.summary())
            models[name] = res1
            results_roc[name], fpr[name], tpr[name], precision[name], recall[name], roc_output[name], precision_output[name], brier_output[name] = estimate_fit(res1 , sm.add_constant(loaded_data["x_test_nl2"], prepend = False), loaded_data["y_test"])
        elif name == "Logit":
            res1 = clf.fit()
            print(res1.summary())
            models[name] = res1
            results_roc[name], fpr[name], tpr[name], precision[name], recall[name], roc_output[name], precision_output[name], brier_output[name] = estimate_fit(res1 , sm.add_constant(loaded_data["x_test"], prepend = False), loaded_data["y_test"] )
        else:
            if "Sigmoid" not in name and "Isotonic" not in name:
                print("Base Calibration")
                clf.fit(loaded_data["x_train"], loaded_data["y_train"])
                results_roc[name], fpr[name], tpr[name], precision[name], recall[name], roc_output[name], precision_output[name], brier_output[name] = estimate_fit(clf, loaded_data["x_test"], loaded_data["y_test"])
            else:
                print("CV Calibration")
                clf.fit(loaded_data["x_cal"], loaded_data["y_cal"])
                results_roc[name], fpr[name], tpr[name], precision[name], recall[name], roc_output[name], precision_output[name], brier_output[name] = estimate_fit(clf, loaded_data["x_test"], loaded_data["y_test"])

        if name == "LogitNonLinear2" or name == "LogitNonLinear" or name == "Logit":                
            with open(path + fn_head + "%s_race%d_interestrate%d.pkl" % (name, race, int_rate), 'wb') as f:
                res1.save(f, remove_data=True)
        else:                
            with open(path + fn_head + "%s_race%d_interestrate%d.pkl" % (name, race, int_rate), 'wb') as f:
                joblib.dump(clf,f)

    with open(path + fn_head + "eval_output_race%d_interestrate%d.csv" % (race, int_rate), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model", "roc", "precision", "brier_score"])
        for name in names:
            writer.writerow([name, roc_output[name], precision_output[name], brier_output[name]])


    return {"models" : models,
            "roc_curve" : {"fpr" : fpr, "tpr" : tpr},
            "precision_curve" : {"precision" : precision, "recall": recall},
            "stats" : [roc_output, precision_output, brier_output] }


def estimate_race_classifier_set(path, loaded_data, race=0, int_rate=0, fn_head = "", additional_model = False):
    if additional_model:
        names = ["Logit",
                 "LogitNonLinear",
                 "LogitNonLinear2",             
                 "RandomForest",
                 "RandomForestIsotonic"
        ]
    else:
        names = ["Logit",
                 "LogitNonLinear",
                 "RandomForest",
                 "RandomForestIsotonic"
        ]
    
    clf_forest = RandomForestClassifier(n_estimators=2,
                                        max_depth=None,
                                        min_samples_split=200,
                                        min_samples_leaf = 100,
                                        random_state=seed,
                                        bootstrap = False,
                                        verbose = 1,
                                        n_jobs=-1)

    if additional_model:
        names = ["Logit",
                 "LogitNonLinear",
                 "LogitNonLinear2",             
                 "RandomForest",
                 "RandomForestIsotonic"
        ]
        classifiers = [ sm.Logit(loaded_data["y_black_train"], sm.add_constant(loaded_data["x_train"], prepend=False)),
                        sm.Logit(loaded_data["y_black_train"], sm.add_constant(loaded_data["x_train_nl"], prepend=False)),
                        sm.Logit(loaded_data["y_black_train"], sm.add_constant(loaded_data["x_train_nl2"], prepend=False)),
                        clf_forest,
                        CalibratedClassifierCV(clf_forest,
                                               method='isotonic',
                                               cv = "prefit")]
    else:
        names = ["Logit",
                 "LogitNonLinear",
                 "RandomForest",
                 "RandomForestIsotonic"
        ]
        classifiers = [ sm.Logit(loaded_data["y_black_train"], sm.add_constant(loaded_data["x_train"], prepend=False)),
                        sm.Logit(loaded_data["y_black_train"], sm.add_constant(loaded_data["x_train_nl"], prepend=False)),
                        clf_forest,
                        CalibratedClassifierCV(clf_forest,
                                               method='isotonic',
                                               cv = "prefit")]
                
    fpr= dict()
    tpr = dict()
    precision= dict()
    recall = dict()
    roc_output = dict()
    precision_output = dict()
    brier_output = dict()
    results_roc = dict()

    models = dict(zip(names, classifiers))
    for name in names:
        clf = models[name]
        print("Fitting %s" % name)
        if name == "LogitNonLinear":
            res1 = clf.fit()
            print(res1.summary())
            models[name] = res1
            y_hat_test = pd.DataFrame(data=res1.predict(sm.add_constant(loaded_data["x_test_nl"], prepend=False)))
            results_roc[name], fpr[name], tpr[name], precision[name], recall[name], roc_output[name], precision_output[name], brier_output[name] = estimate_fit(y_hat_test, loaded_data["y_black_test"])
        elif name == "LogitNonLinear2":
            res1 = clf.fit()
            print(res1.summary())
            models[name] = res1
            y_hat_test = pd.DataFrame(data=res1.predict(sm.add_constant(loaded_data["x_test_nl2"], prepend=False)))
            results_roc[name], fpr[name], tpr[name], precision[name], recall[name], roc_output[name], precision_output[name], brier_output[name] = estimate_fit(y_hat_test, loaded_data["y_black_test"])
        elif name == "Logit":
            res1 = clf.fit()
            print(res1.summary())
            models[name] = res1
            y_hat_test = pd.DataFrame(data=res1.predict(sm.add_constant(loaded_data["x_test"], prepend=False)))
            results_roc[name], fpr[name], tpr[name], precision[name], recall[name], roc_output[name], precision_output[name], brier_output[name] = estimate_fit(y_hat_test, loaded_data["y_black_test"])
        else:
            if "Sigmoid" not in name and "Isotonic" not in name:
                print("Base Calibration")
                clf.fit(loaded_data["x_train"], loaded_data["y_black_train"])
                y_hat_test = pd.DataFrame(data=clf.predict_proba(loaded_data["x_test"]))[1]                
                results_roc[name], fpr[name], tpr[name], precision[name], recall[name], roc_output[name], precision_output[name], brier_output[name] = estimate_fit(y_hat_test, loaded_data["y_black_test"])
            else:
                print("CV Calibration")
                clf.fit(loaded_data["x_cal"], loaded_data["y_black_cal"])
                y_hat_test = pd.DataFrame(data=clf.predict_proba(loaded_data["x_test"]))[1]                
                results_roc[name], fpr[name], tpr[name], precision[name], recall[name], roc_output[name], precision_output[name], brier_output[name] = estimate_fit(y_hat_test, loaded_data["y_black_test"])
        
        if name == "LogitNonLinear2" or name == "LogitNonLinear" or name == "Logit":                
            with open(path + fn_head + "race_%s_race%d_interestrate%d.pkl" % (name, race, int_rate), 'wb') as f:
                res1.save(f, remove_data=True)
        else:
            with open(path + fn_head + "race_%s_race%d_interestrate%d.pkl" % (name, race, int_rate), 'wb') as f:
                joblib.dump(clf,f)

    with open(path + fn_head + "eval_race_output_race%d_interestrate%d.csv" % (race, int_rate), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model", "roc", "precision", "brier_score"])
        for name in names:
            writer.writerow([name, roc_output[name], precision_output[name], brier_output[name]])


    return {"models" : models,
            "roc_curve" : {"fpr" : fpr, "tpr" : tpr},
            "precision_curve" : {"precision" : precision, "recall": recall},
            "stats" : [roc_output, precision_output, brier_output] }


def plot_roc(path="", graph_data={}, graph_labels={}, race = 0, int_rate = 0, fn_head = ""):
    names_graph = graph_labels.keys()
    fig, ax = plt.subplots()
    for name in names_graph:
        try: 
            fpr = graph_data["fpr"][name]
            tpr = graph_data["tpr"][name]
            plt.plot(fpr, tpr,
                     label='%s' % graph_labels[name])
        except KeyError:
            print("Model %s not in model list" % name )
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.legend(loc="best", fontsize=20)
    for tick in fig.get_axes()[0].get_xticklabels():
        tick.set_fontsize(20)
    for tick in fig.get_axes()[0].get_yticklabels():
        tick.set_fontsize(20)
    plt.savefig(path + fn_head + "roc_auc_big_race%d_interestrate%d.pdf" % (race, int_rate), bbox_inches='tight')
    plt.close(1)

def plot_precision(path = "output/", graph_data={}, graph_labels={}, race = 0, int_rate = 0, fn_head = ""):
    names_graph = graph_labels.keys()
    fig, ax = plt.subplots()
    for name in names_graph:
        try:
            recall = graph_data["recall"][name]
            precision = graph_data["precision"][name]
            ax = plt.plot(recall, precision,
                          label='%s ' % graph_labels[name])
        except KeyError:
            print("Model %s not in model list" % name )
            
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall',fontsize=20)
    plt.ylabel('Precision',fontsize=20)
    plt.legend(loc="best",fontsize=20)
    for tick in fig.get_axes()[0].get_xticklabels():
        tick.set_fontsize(20)
        for tick in fig.get_axes()[0].get_yticklabels():
            tick.set_fontsize(20)
    plt.savefig(path + fn_head + "precision_recall_big_race%d_interestrate%d.pdf" % (race, int_rate), bbox_inches='tight')
    plt.close()


def plot_calibration(path = "output/", loaded_data_norace={}, estimated_test_probs={}, graph_labels={}, race = 0, int_rate = 0, fn_head = ""):
    fraction_of_positives = {}
    mean_predicted_value = {}
    for name in graph_labels.keys():
        try:
            fraction_of_positives[name], mean_predicted_value[name] =  calibration_curve(
                pd.to_numeric(loaded_data_norace["y_test"], errors='coerce'),
                pd.to_numeric(estimated_test_probs[name], errors='coerce'), n_bins=20)
        except KeyError:
            print("Model %s not in model list" % name )

    fig, ax1 = plt.subplots()
    ax1.set_ylabel("Fraction of positives",fontsize=20)
    ax1.set_xlabel("Mean predicted value",fontsize=20)
    ax1.set_ylim([-0.05, 1.05])

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")    
    for name in graph_labels.keys():
        try:
            ax1.plot(mean_predicted_value[name], fraction_of_positives[name], "s-",
                     label="%s" % (graph_labels[name]))
        except KeyError:
            print("Model %s not in model list" % name )
    
    ax1.legend(loc="best",fontsize=20)
    for tick in fig.get_axes()[0].get_xticklabels():
        tick.set_fontsize(20)
    for tick in fig.get_axes()[0].get_yticklabels():
        tick.set_fontsize(20)
    plt.tight_layout()
    plt.savefig(path + fn_head + "calibration_race%d_interestrate%d.pdf" % (race, int_rate), bbox_inches='tight')
    plt.close(1)

    
    
#the “race” input is just a pandas series that has the cleaned up race values from above
def cdf_pd_diff(df,race,plotrace,clfs,lim = (-1,1),log=False):
    outnames = {"LogitNonLinear": "Nonlinear Logit","LogitNonLinear2": "Nonlinear Logit (Full)", "RandomForestIsotonic" : "Random Forest","xgboost_output": "XGBoost","Difference" : "Difference"}
    fig, ax = plt.subplots(1,1,figsize=(7.5,5))
    if log:
        diff = df[clfs[1]].apply(np.log) - df[clfs[0]].apply(np.log)
        factor = 1
    else:
        diff = df[clfs[1]] - df[clfs[0]]
        factor = 100
    for group in plotrace:
        x = np.sort(factor*(diff[race==group]))
        y = np.linspace(1,x.shape[0],x.shape[0]) / x.shape[0]
        ax.plot(x,y,label=group)
    ax.set_xlim(lim); ax.set_ylim((0,1))
    ax.set_xticks([-1,-0.5,0,0.5,1])
#     ax.set_yticks([0,0.25,0.5,0.75,1])
    ax.axvline(0,color='k')
    ax.axhline(0.5,color='k',linestyle='--')
    plt.xticks(fontsize=16); plt.yticks(fontsize = 16)
    if log: 
        ax.set_xlabel('Log(PD from %s) - Log(PD from %s)' %(outnames[clfs[1]],outnames[clfs[0]]),fontsize = 18)
    else:
        ax.set_xlabel('PD from %s - PD from %s' %(outnames[clfs[1]],outnames[clfs[0]]),fontsize = 18)
    ax.set_ylabel('Cumulative Share',fontsize=18)
    ax.legend(frameon=True,framealpha=1,fontsize = 16,loc='lower right')


def bootstrap_models(path, loaded_data, models, k = 500, additional_model = False):
    # take in data and estimated model
    # permute test sample 500 times
    # extract statistic each time
    # Compare just rf nonlinear and random forests

    if additional_model:
        names = ["Logit",
                 "LogitNonLinear",
                 "LogitNonLinear2",             
                 "RandomForest",
                 "RandomForestIsotonic"
        ]
    else:
        names = ["Logit",
                 "LogitNonLinear",
                 "RandomForestIsotonic"
        ]

    stats = []
    y_test = {}
    y_test["LogitNonLinear"] = pd.DataFrame(data=models["LogitNonLinear"].predict(sm.add_constant(loaded_data["x_test_nl"], prepend=False)))
    y_test["Logit"] = pd.DataFrame(data=models["Logit"].predict(sm.add_constant(loaded_data["x_test"], prepend=False)))
    y_test["RandomForestIsotonic"] = pd.DataFrame(data=models["RandomForestIsotonic"].predict_proba(loaded_data["x_test"]))[1]
    
    for i in range(k):
        if i % 50 == 0:
            print(i)
        n = len(loaded_data["x_test"])
        idx = np.random.randint(n, size=n)
        fpr= dict()
        tpr = dict()
        precision= dict()
        recall = dict()
        roc_output = dict()
        precision_output = dict()
        brier_output = dict()
        results_roc = dict()
        for name in names:
            clf = models[name]
            print("Testing %s" % name)
            results_roc[name], fpr[name], tpr[name], precision[name], recall[name], roc_output[name], precision_output[name], brier_output[name] = estimate_fit(y_test[name].iloc[idx], loaded_data["y_test"].iloc[idx])
        stats.append([roc_output, precision_output, brier_output])

    roc_output = {}
    roc_output["Logit"] = [stat[0]["Logit"] for stat in stats]
    roc_output["LogitNonLinear"] = [stat[0]["LogitNonLinear"] for stat in stats]
    roc_output["RandomForestIsotonic"] = [stat[0]["RandomForestIsotonic"] for stat in stats]    
    precision_output = {}
    precision_output["Logit"] = [stat[1]["Logit"] for stat in stats]
    precision_output["LogitNonLinear"] = [stat[1]["LogitNonLinear"] for stat in stats]
    precision_output["RandomForestIsotonic"] = [stat[1]["RandomForestIsotonic"] for stat in stats]    
    brier_output = {}
    brier_output["Logit"] = [stat[2]["Logit"] for stat in stats]
    brier_output["LogitNonLinear"] = [stat[2]["LogitNonLinear"] for stat in stats]
    brier_output["RandomForestIsotonic"] = [stat[2]["RandomForestIsotonic"] for stat in stats]    

    with open(path + "roc_output_bootstrap.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(roc_output.keys())
        writer.writerows(map(list, zip(*roc_output.values())))

    plt_data = map(list, zip(*roc_output.values()))
    plt_data = [x[2] - x[1] for x in plt_data]
    ax = plt.hist(plt_data)
    #plt.xlim([0, 1])
    plt.xlabel('Random Forest - Nonlinear Logit',fontsize=20)
    plt.savefig(path + "roc_bootstrap.pdf", bbox_inches='tight')
    plt.close()
        
    with open(path + "precision_output_bootstrap.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(precision_output.keys())
        writer.writerows(map(list, zip(*precision_output.values())))

    plt_data = map(list, zip(*precision_output.values()))
    plt_data = [x[2] - x[1] for x in plt_data]
    ax = plt.hist(plt_data)
    #plt.xlim([0, 1])
    plt.xlabel('Random Forest - Nonlinear Logit',fontsize=20)
    plt.savefig(path + "precision_bootstrap.pdf", bbox_inches='tight')
    plt.close()
        
    with open(path + "brier_output_bootstrap.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(brier_output.keys())
        writer.writerows(map(list, zip(*brier_output.values())))

    plt_data = map(list, zip(*brier_output.values()))
    plt_data = [100*(x[1] - x[2]) for x in plt_data]
    ax = plt.hist(plt_data)
    #plt.xlim([0, 1])
    plt.xlabel('(Nonlinear Logit - Random Forest) x 100',fontsize=20)
    plt.savefig(path + "brier_bootstrap.pdf", bbox_inches='tight')
    plt.close()

    plt_data = map(list, zip(*brier_output.values()))
    incidence = np.mean(loaded_data["y_test"])
    plt_data = [[ 1 - (x / ((1-incidence)*incidence)) for x in y] for y in plt_data]
    plt_data = [x[2] - x[1] for x in plt_data]
    ax = plt.hist(plt_data)
    #plt.xlim([0, 1])
    plt.xlabel('Random Forest - Nonlinear Logit',fontsize=20)
    plt.savefig(path + "r2_bootstrap.pdf", bbox_inches='tight')
    plt.close()
    
    return roc_output, precision_output, brier_output


def bootstrap_models_resample(path, loaded_data, feature_names_norace, int_rate, k = 50,  additional_model = False):
    # take in data
    # permute data
    # estimate model
    # Compare 

    if additional_model:
        names = ["Logit",
                 "LogitNonLinear",
                 "LogitNonLinear2",             
                 "RandomForest",
                 "RandomForestIsotonic"
        ]
    else:
        names = ["Logit",
                 "LogitNonLinear",
                 "RandomForestIsotonic"
        ]

    stats = []
    
    for i in range(k):
        if i % 50 == 0:
            print(i)
        # reshuffled_data = scramble_datasets(loaded_data["full_data"],
        #                                     feature_names_norace,
        #                                     int_rate)
        reshuffled_data = bootstrap_datasets(loaded_data,
                                            feature_names_norace,
                                            int_rate)
        fpr= dict()
        tpr = dict()
        precision= dict()
        recall = dict()
        roc_output = dict()
        precision_output = dict()
        brier_output = dict()
        results_roc = dict()
        estimation_output_norace_bs = estimate_classifier_set(reshuffled_data,
                                                              0,
                                                              int_rate,
                                                              fn_head = "bootstrap" + str(i),
                                                              additional_model = False,
                                                              save_model = False)
        stats.append(estimation_output_norace_bs["stats"])

    roc_output = {}
    roc_output["Logit"] = [stat[0]["Logit"] for stat in stats]
    roc_output["LogitNonLinear"] = [stat[0]["LogitNonLinear"] for stat in stats]
    roc_output["RandomForestIsotonic"] = [stat[0]["RandomForestIsotonic"] for stat in stats]    
    precision_output = {}
    precision_output["Logit"] = [stat[1]["Logit"] for stat in stats]
    precision_output["LogitNonLinear"] = [stat[1]["LogitNonLinear"] for stat in stats]
    precision_output["RandomForestIsotonic"] = [stat[1]["RandomForestIsotonic"] for stat in stats]    
    brier_output = {}
    brier_output["Logit"] = [stat[2]["Logit"] for stat in stats]
    brier_output["LogitNonLinear"] = [stat[2]["LogitNonLinear"] for stat in stats]
    brier_output["RandomForestIsotonic"] = [stat[2]["RandomForestIsotonic"] for stat in stats]    

    with open(path + "roc_output_bootstrap_resample.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(roc_output.keys())
        writer.writerows(map(list, zip(*roc_output.values())))

    plt_data = map(list, zip(*roc_output.values()))
    plt_data = [x[2] - x[1] for x in plt_data]
    ax = plt.hist(plt_data)
    #plt.xlim([0, 1])
    plt.xlabel('Random Forest - Nonlinear Logit',fontsize=20)
    plt.savefig(path + "roc_bootstrap_resample.pdf", bbox_inches='tight')
    plt.close()
        
    with open(path + "precision_output_bootstrap_resample.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(precision_output.keys())
        writer.writerows(map(list, zip(*precision_output.values())))

    plt_data = map(list, zip(*precision_output.values()))
    plt_data = [x[2] - x[1] for x in plt_data]
    ax = plt.hist(plt_data)
    #plt.xlim([0, 1])
    plt.xlabel('Random Forest - Nonlinear Logit',fontsize=20)
    plt.savefig(path + "precision_bootstrap_resample.pdf", bbox_inches='tight')
    plt.close()
        
    with open(path + "brier_output_bootstrap_resample.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(brier_output.keys())
        writer.writerows(map(list, zip(*brier_output.values())))

    plt_data = map(list, zip(*brier_output.values()))
    plt_data = [100*(x[1] - x[2]) for x in plt_data]
    ax = plt.hist(plt_data)
    #plt.xlim([0, 1])
    plt.xlabel('(Nonlinear Logit - Random Forest) x 100',fontsize=20)
    plt.savefig(path + "brier_bootstrap_resample.pdf", bbox_inches='tight')
    plt.close()

    plt_data = map(list, zip(*brier_output.values()))
    incidence = np.mean(loaded_data["y_test"])
    plt_data = [[ 1 - (x / ((1-incidence)*incidence)) for x in y] for y in plt_data]
    plt_data = [x[2] - x[1] for x in plt_data]
    ax = plt.hist(plt_data)
    #plt.xlim([0, 1])
    plt.xlabel('Random Forest - Nonlinear Logit',fontsize=20)
    plt.savefig(path + "r2_bootstrap_resample.pdf", bbox_inches='tight')
    plt.close()
    
    return roc_output, precision_output, brier_output



