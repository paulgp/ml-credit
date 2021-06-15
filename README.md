
# Document for Replicating "Predictably Unequal? The Effects of Machine Learning on Credit Markets"

The structure of this document first outlines where the output of each exhibit in the paper is constructed. These exihibits are constructed from a main analysis file, ``all_vals_race1_interestrate1.csv.`` However, we cannot provide this file, as it is proprietary data. As a result, we provide a simulated version of the dataset to allow execution of the code. 

At the end of the document in the Clean Data section, we list the relevant code and files necessary to run to generate ``all_vals_race1_interestrate1.csv`` from the Federal Reserve databases.

## Exhibits in Paper. 
See discussion below for each program's relevant inputs and outputs
	
### Main Draft

1. TABLE I - Descriptive Stats
   * ``descriptive_stats.do``
2. FIGURE 3 - ROC and Precision-Recall Curves
   * Constructed in ``run_estimation_programs.py``
3. TABLE III - Output of different models
   * Estimated in ``run_estimation_programs.py`` and converted into a table using ``make_tables_draft_01142020.do``
4. TABLE IV - Output of different models predicting race
   * Estimated in ``run_estimation_programs.py`` and converted into a table using ``make_tables_draft_01142020.do``
5. FIGURE IV - Example of Predicted Default Probabilities
   * Uses model from ``run_estimation_programs.py`` and constructed in ``replicate_fig4_fig5_tabVI.ipynb``
6. FIGURE V - Comparison of Predicted Default Probabilities Across Models, by Race Groups
   * Uses model from ``run_estimation_programs.py`` and constructed in ``replicate_fig4_fig5_tabVI.ipynb``
7. TABLE VI - Decomposition of Performance Improvement
   * uses model from ``run_estimation_programs.py`` and constructed in ``make_tables_draft_01142020.do``
8. TABLE VII - Equilibrium Outcomes
   * uses model from ``run_estimation_programs.py`` and constructed in ``replicate_tabVII.ipynb``

### Internet Appendix
9. Figure IA.2 Calibration Curve
	* Construct in ``run_estimation_programs.py`` 
12. Table IA.II Descriptive Statistics, GSE, Full Documentation Originations.
	* Constructed in ``descriptive_stats.do``
13. Table IA.III Residual Variation in SATO, comparing Full and Equilibrium samples.
	* Constructed in ``descriptive_stats.do``
14. Table IA.IV. Decomposition of Equilibrium Effects
	
15. Figure IA.3 Comparison of Predicted Default Probabilities — XGBoost vs. Nonlinear Logit
	* This output is constructed using ``xg_boost.R`` The output is then fed back into ``run_estimation_programs.py``
16. Figure IA.4. Residual interest rate variation.
	* This Figure is constructed using the output in Table IA.4
17. Figure IA.5. Comparison of Equilibrium Interest Rates
	
18. Table IA.V. Equilibrium Effects Under Alternative Approach
	* This Table is constructed using table_IV_V.do
19. Figure IA.6. Bootstrap Estimates of Differences in Statistics
	* Construct in ``run_estimation_programs.py`` -- WARNING: this takes a very long time to run.	

-----------

## Estimation Code

### Stata Code
1.  ``descriptive_stats.do``

	This do-file creates Table 1, Table IA.II, and  Table IA.III 
    * Input data:
	1. ``all_vals_race1_interestrate1.csv``

2. ``make_tables_draft_01142020.do``
	This do-file creates Table III, IV and VI
   
3. ``logits_nov2019.do``
	This do-file does the full interaction logit for Column 8 of Table V

### Python Code
1.  ``run_estimation_programs.py``

	This Python code runs programs from `estimation_programs.py`. It produces a significant amount of output in the output/ folder. [The path is hardcoded, so make sure you have an output folder available.] Importantly, this code has a number of very slow programs (estimation, especially with bootstrapping). Ideally, you only need to run these once, and then they are pickled. 

	* Input data:
		1. ``sato_varnames_race1.csv`` and ``sato_varnames_race0.csv``
		2. ``all_vals_race1_interestrate1.csv``

2.  ``replicate_fig4_fig5_tabVI.ipynb`` and  ``replicate_tabVII.ipynb``

	These are iPython notebooks. 
	* Input data::
		1.sato_varnames_race%d.csv, %d = race
		2.all_vals_race1_interestrate1.csv
		  all_vals_race0_interestrate1.csv	 
	    3.predictions_race0_interestrate0.csv
		4.%s_race0_interestrate1.pkl, %s = model name (Logit, LogitNonlinear, RandomForestIsotonic)
		5.%s_race0_interestrate0.pkl, %s = model name (Logit, LogitNonlinear, RandomForestIsotonic)
	 

----------------

## Clean Data

N.B. This code is not executable with the given files b/c we do not
provide a simulated version of the initial files (just the main
analysis file). We provide this code in case researchers with access
to the Federal Reserve RADAR system are interested in constructing
this code.

** Prep data
1.  ``data_pull_clean.do`` (run on Federal Reserve RADAR system)

	This code has two parts. First is a set of SQL code for the data
    pull to construct the McDash static and dynamic data (these are
    commented out at the beginning). The second portion constructs our
    main file from these pulls. These CSV files are then stitched
    together and constructed into the main
    ``all_vals_race1_interestrate1.csv`` data
	

	

	
