# House Price Prediction

> This project implements machine learning models to predict house prices using the Ames Housing dataset. The implementation includes a comprehensive data preprocessing pipeline, model training, evaluation, and hyperparameter tuning to achieve optimal prediction results.

## Dataset
The dataset contains information about residential homes in Ames, Iowa, with 79 explanatory variables describing various aspects of the houses:
- [train.csv](data/train.csv): Training data with 1460 observations and includes the target variable `SalePrice`
- [test.csv](data/test.csv): Test data with 1459 observations used for making predictions
- [data_description.txt](data/data_description.txt): Detailed description of all variables in the dataset

## Project Structure
- `testmodel.ipynb`: Main notebook with the complete modeling pipeline
- `housepriceprediction.ipynb`: Additional exploratory notebook
- `percobaan.ipynb`: Notebook for experimental approaches
- `submission.csv`: Predictions file in the format required for submission
- `requirements.txt`: Python dependencies required for the project
- `data/`: Directory containing all dataset files

## Methodology

### Data Preprocessing
The preprocessing pipeline implemented in [`preprocess_house_data`](testmodel.ipynb) function includes:
1. Handling missing values with different strategies based on variable type and missing percentage
2. Feature transformation (logarithmic, Yeo-Johnson) for skewed numerical variables
3. Categorical encoding with ordinal mapping based on target relationship
4. Feature engineering including temporal variable transformations
5. Feature selection using Lasso regularization

### Model Development
The project evaluates multiple regression models:
- Linear models: Linear Regression, Ridge, Lasso, ElasticNet
- Tree-based models: Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
- Other models: SVR, KNN

Models are evaluated using:
- Cross-validation with 5 folds
- Metrics: RMSE, MSE, and R²
- Visualization of comparative performance

### Model Optimization
- Hyperparameter tuning using GridSearchCV
- Ensemble modeling with the best-performing models

## Key Functions
- [`preprocess_house_data`](testmodel.ipynb): Comprehensive data preprocessing pipeline
- [`evaluate_model`](testmodel.ipynb): Model training and evaluation on train/test split
- [`cross_val_evaluate`](testmodel.ipynb): K-fold cross-validation evaluation
- [`ensemble_predict`](testmodel.ipynb): Ensemble prediction function

## Results
The best models after optimization include XGBoost, LightGBM, and Gradient Boosting. The final solution uses an ensemble approach, averaging predictions from multiple tuned models to achieve robust results.

## Running the Project
1. Install dependencies:
```bash 
pip install -r requirements.txt
```
2. Run the Jupyter notebooks:
```bash
jupyter notebook testmodel.ipynb
```
2. or for EDA:
```bash
jupyter notebook housepriceprediction.ipynb
```

## License
This project is open source and available for educational and research purposes.