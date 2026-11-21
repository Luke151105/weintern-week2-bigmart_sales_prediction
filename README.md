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

Business Interpretation of the Prediction Model

The tuned XGBoost model with 13 selected features achieved the best performance among the evaluated models, with an R² of 0.5811, MAE of 800.01, and RMSE of 1137.24.

The model can support retail decision-making by providing estimated sales figures for different products and outlets. These estimates may help with inventory planning, product allocation, and understanding outlet-level sales patterns.

However, the model does not explain all variations in sales, and its predictions should be treated as estimates rather than exact sales figures.
  
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


Visualizations:

Numerical Features:

Item Weight 

<img width="397" height="374" alt="Item Weight Distribution" src="https://github.com/user-attachments/assets/6c4b2966-554f-4bb7-ab5a-79c2ed1557ab" />

Item Visibility 

<img width="391" height="374" alt="Item Visibility Distribution" src="https://github.com/user-attachments/assets/d193f3ce-b62a-41d7-bb4b-d1b12401145d" />

Item MRP

<img width="391" height="374" alt="Item MRP Distribution" src="https://github.com/user-attachments/assets/b7dcda2f-cb08-4bc2-a646-1121584b6cec" />

Item_Outlet_Sales

<img width="391" height="374" alt="Item Outlet Sales Distribution" src="https://github.com/user-attachments/assets/47ca7f00-6202-4486-b8bc-a5a2699c0dd6" />

Outlet Establishment Year

<img width="397" height="374" alt="Outlet Establishment Year" src="https://github.com/user-attachments/assets/5d9b7be0-6b42-4d9f-ae29-2526d1352297" />


Categorical Features:

Item Type
<img width="1736" height="374" alt="Items by Item Type" src="https://github.com/user-attachments/assets/d952daa6-690e-47f1-912c-46368186f2c9" /> <img width="363" height="231" alt="Item Type Share Pie Chart" src="https://github.com/user-attachments/assets/2020a496-6453-48c8-9ff4-fcbac2943f4e" />


Outlet Size 

<img width="397" height="375" alt="Outlet Size Distribution" src="https://github.com/user-attachments/assets/49636d75-ccf1-4c62-94b7-8b560742cfc3" />

Outlet Location type

<img width="397" height="376" alt="Outlet Location Type Distribution" src="https://github.com/user-attachments/assets/470e0636-79a2-44f0-aed1-90e49635ec44" />

Outlet Type

<img width="1736" height="374" alt="Outlet Type Distribution" src="https://github.com/user-attachments/assets/8955d167-655b-4cff-9861-093996214088" />

Item Fat Content

<img width="397" height="374" alt="Item Fat Content Distribution" src="https://github.com/user-attachments/assets/680c00ee-f45c-4eef-a1f8-2a82ef394046" />

Bivariate Analysis

Sales by Item Type 

<img width="1457" height="863" alt="Average Sales by Item Type" src="https://github.com/user-attachments/assets/5924b038-5ed2-437e-92d0-eb93ec90c74b" />

Sales by Item Visibility 

<img width="514" height="319" alt="Item Visibility vs Sales" src="https://github.com/user-attachments/assets/49a6fff6-009c-4800-aa5a-e9ddae6a65f0" />

Sales by Item MRP

<img width="1170" height="305" alt="Item MRP vs Sales" src="https://github.com/user-attachments/assets/822f8883-cee7-4e86-985b-dc3f5832e04a" />

Sales by Outlet Size

<img width="397" height="265" alt="Average Sales by Outlet Size" src="https://github.com/user-attachments/assets/68fa2fca-a204-4940-ae54-784797471d1a" />


Sales by Outlet Location Type

<img width="397" height="265" alt="Average Sales by Outlet Location Type" src="https://github.com/user-attachments/assets/33b332fe-ac58-4e52-89f7-24923f83c9ff" />


Sales by Outlet Type

<img width="508" height="268" alt="Average Sales by Outlet Type" src="https://github.com/user-attachments/assets/1ffa05c5-df95-40c8-b368-da0be225bd10" />


Correlation Heatmap 

<img width="683" height="704" alt="Correlation Heatmap" src="https://github.com/user-attachments/assets/b4dd205b-4a5c-48a8-bd9d-09ce36c50e6b" />


Actual vs Predicted values for Linear Regression 

<img width="410" height="388" alt="Actual vs Predicted - Initial XGBoost" src="https://github.com/user-attachments/assets/d4afcfda-7741-4130-9c80-0a11a3b9b131" />


Actual vs Predicted values for XG Boost model using feature importance

<img width="404" height="388" alt="Actual vs Predicted - Linear Regression" src="https://github.com/user-attachments/assets/f75703d7-9b43-4906-84f9-434211a8d56f" />



Actual vs Predicted values for XG Boost model using hyperparametric tuning 

<img width="404" height="388" alt="Actual vs Predicted - Tuned XGBoost" src="https://github.com/user-attachments/assets/2436d14a-ec4c-415a-9cef-2605f7707c03" />












