Big Mart Sales Prediction 

Project overview

This project analyses the BigMart retail sales dataset and develops regression models to estimate Item_Outlet_Sales using product and outlet characteristics. The work is organised into data cleaning, exploratory data analysis (EDA), and sales prediction.

Dataset
- Input file expected by the notebook: Train.csv
- Observations: 8,523
- Original columns: 12
- Prediction target: Item_Outlet_Sales

The notebook uses these fields: Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type, and Item_Outlet_Sales.

Workflow

1. Data cleaning
- Checked duplicate rows; none were found in the loaded dataset.
- Filled missing Item_Weight values using the mean weight within each Item_Type.
- Filled missing Outlet_Size values using the most frequent outlet size within each Outlet_Type.
- Standardised Item_Fat_Content labels (low fat and LF to Low Fat; reg to Regular).
  
2. Exploratory data analysis
  Performed descriptive statistics, numeric distributions, categorical counts, relationships between features and sales, and a correlation heatmap. Visualisations include histograms/KDE plots, count plots, a pie chart, bar plots, and scatter plots.

Observations:

- Item_MRP has a positive correlation with Item_Outlet_Sales (approximately 0.568).
- Item_Visibility has a weak negative correlation with Item_Outlet_Sales (approximately -0.129).
- The strongest feature importances in the initial XGBoost model were Outlet_Identifier_OUT027 and Outlet_Type_Supermarket Type1.
  
3. Sales prediction
- Removed Item_Identifier from the predictors.
- Applied one-hot encoding with pd.get_dummies(..., drop_first=True), producing 37 columns in the encoded dataframe (including the target).
- Used an 80:20 train-test split with random_state=2.
- Trained Linear Regression as a baseline and XGBoost Regressor as the tree-based model.
- Ranked feature importances from the initial XGBoost model, selected 13 predictors, and retrained XGBoost.
- Tuned the 13-feature XGBoost model using RandomizedSearchCV with 20 sampled parameter combinations and 5-fold cross-validation.
  
<img width="1752" height="897" alt="image" src="https://github.com/user-attachments/assets/abc50192-c3c1-473a-9e39-28124aa8bddc" />



The tuned XGBoost model is the best of the evaluated models on the held-out test split according to all three reported metrics.
Best hyperparameters found

- colsample_bytree: 0.91196
- learning_rate: 0.10139
- max_depth: 3
- n_estimators: 116
- subsample: 0.82781
  
Requirements

The notebook uses:
- Python 3
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- SciPy
  
Install dependencies with:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy jupyter

How to run

1. Place Train.csv in the same folder as the notebook.
2. Install the dependencies listed above.
3. Open BigMart Sales Prediction Using Machine Learning.ipynb in Jupyter Notebook or JupyterLab.
4. Run the notebook cells from top to bottom.
   
Important methodological notes

- The notebook performs missing-value imputation and one-hot encoding before the train-test split. For a more rigorous production workflow, fit imputation and encoding steps using training data only, ideally within a scikit-learn Pipeline/`ColumnTransformer`.
- The notebook's feature selection uses importances from an XGBoost model trained on the training split, then evaluates the selected-feature model on a new 80:20 split with the same random seed.
- No explicit outlier-removal or outlier-capping step is present. The EDA includes distribution plots, but inspection is not the same as treatment.
- The dataset does not contain a transaction date field, so date-based lag or rolling-window features were not created.
- Feature importance indicates predictive contribution within this fitted model; it does not establish causation.
  
  
Files

- BigMart Sales Prediction Using Machine Learning.ipynb — analysis and modelling notebook
- Train.csv — input dataset (not included in this README file)
- BigMart_Sales_Prediction_Report.pdf / .docx — project report
