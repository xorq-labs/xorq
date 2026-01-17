# Available Libraries Reference

You have access to a rich set of libraries for various problem types. Choose the right tool for the job!

## 🔬 Scientific Computing & Optimization
- **scipy**: Optimization (minimize, linprog), statistics, signal processing, interpolation
  - Use for: Portfolio optimization, curve fitting, statistical tests, linear programming
- **numpy**: Array operations, linear algebra, mathematical functions
  - Use for: Numerical computations, array manipulations, matrix operations
- **pulp**: Linear and integer programming
  - Use for: Resource allocation, scheduling, production planning, assignment problems

## 📈 Statistical Modeling & Time Series
- **statsmodels**: Statistical models, time series analysis, regression
  - Use for: ARIMA, SARIMA, VAR models, linear regression, GLM, hypothesis testing
- **prophet**: Time series forecasting (Facebook Prophet)
  - Use for: Business forecasting with seasonality, trend analysis, holiday effects

## 🤖 Machine Learning (via xorq)
- **sklearn**: Classification, regression, clustering, preprocessing
  - Available through xorq's deferred execution: LogisticRegression, RandomForest, etc.
  - Use with: `Pipeline.from_instance()` for deferred ML pipelines
- **xgboost**: Gradient boosting algorithms
  - Use for: High-performance classification/regression with structured/tabular data
  - Works with xorq pipelines via `Pipeline.from_instance()`

## 📊 Visualization & Analysis
- **matplotlib**: Core plotting library
  - Use for: Line plots, scatter plots, histograms, customized visualizations
- **seaborn**: Statistical data visualization
  - Use for: Distribution plots, correlation heatmaps, categorical plots
- **pandas** (in UDFs only!): DataFrames and Series operations
  - CRITICAL: Use only inside `@make_pandas_udf` decorators!

## 🗄️ Data Processing
- **xorq**: Deferred data processing (your primary tool!)
  - Use for: All data transformations, filtering, aggregations, joins
  - Custom logic: Wrap in `@make_pandas_udf` or `@make_pandas_udaf`

## 💡 LIBRARY SELECTION GUIDE

### For Optimization Problems:
- Simple linear programming → `pulp` or `scipy.optimize.linprog`
- Nonlinear optimization → `scipy.optimize.minimize`
- Constrained optimization → `scipy.optimize` with constraints
- Portfolio optimization → `scipy.optimize` (Sharpe ratio, mean-variance)
- ⚠️ ALL optimization code must be wrapped in UDFs!

### For Time Series:
- Business forecasting with seasonality/holidays → `prophet` wrapped in UDF
  - Example: @make_pandas_udf for Prophet().fit() and .predict()
  - Best for: Daily/weekly data with strong seasonality, holiday effects
- Classical time series (ARIMA, SARIMA, VAR) → `statsmodels` in UDF
  - Best for: Stationary series, econometric analysis, multivariate time series
- Simple moving averages/exponential smoothing → Pandas rolling in UDF or ibis window functions
- Trend analysis → `scipy.signal` for detrending, smoothing

### For Statistical Analysis:
- Hypothesis tests, confidence intervals → `scipy.stats`
- Regression models, GLM → `statsmodels`
- Distribution fitting → `scipy.stats`

### For Machine Learning:
- Classification/Regression (balanced data) → `sklearn.linear_model`, `sklearn.ensemble`
- Classification/Regression (large/imbalanced data) → `xgboost.XGBClassifier`, `xgboost.XGBRegressor`
- All sklearn/xgboost models → Use with xorq's `Pipeline.from_instance()`
- Feature preprocessing → `sklearn.preprocessing` with ColumnTransformer (use tuples!)
- Model evaluation → `sklearn.metrics` with `ml.deferred_sklearn_metric()`
- Cross-validation → `xorq.expr.ml` utilities

### For Visualization:
- Quick plots → `matplotlib.pyplot`
- Statistical plots → `seaborn` (heatmaps, distributions)
- Execute data first, then visualize (`.execute()` then plot)
