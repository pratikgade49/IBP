from typing import List, Dict, Tuple, Union
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
import logging

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('algorithms.log')
    ]
)
logger = logging.getLogger(__name__)

def validate_and_clean(values: List[Union[float, str]]) -> List[Union[float, str]]:
    """Replace NaN/inf values with 'NULL' and log issues"""
    cleaned = []
    for i, v in enumerate(values):
        if isinstance(v, (float, int)) and not np.isfinite(v):
            logger.warning(f"Invalid value detected at position {i}: {v}. Replacing with 'NULL'")
            cleaned.append("NULL")
        else:
            cleaned.append(v)
    return cleaned

def safe_float_convert(str_list: List[str]) -> List[float]:
    """Safely convert string list to floats, handling NULLs and invalid values"""
    converted = []
    non_null_values = []

    # First pass: collect valid numeric values
    for s in str_list:
        if s == "NULL":
            continue
        try:
            val = float(s)
            if np.isfinite(val):
                non_null_values.append(val)
        except ValueError:
            pass

    # Calculate average of non-null values or default to 0
    avg = sum(non_null_values) / len(non_null_values) if non_null_values else 0.0

    # Second pass: convert values
    for s in str_list:
        if s == "NULL":
            converted.append(avg)
            logger.warning("Replaced NULL with average value")
        else:
            try:
                val = float(s)
                if np.isfinite(val):
                    converted.append(val)
                else:
                    converted.append(avg)
                    logger.warning(f"Replaced non-finite value {val} with average")
            except ValueError:
                converted.append(avg)
                logger.warning(f"Could not convert '{s}' to float, used average")

    return converted

def average_calculation(planning_object_data: Dict, parameters: Dict, historical_periods: int, forecast_periods: int) -> Dict:
    """Calculates Average Forecast"""
    logger.debug(f"Starting average_calculation with parameters: {parameters}")
    try:
        demand = planning_object_data["HISTORY"]
        logger.debug(f"Historical demand data: {demand}")

        mean_value = sum(demand) / historical_periods
        logger.info(f"Calculated mean value: {mean_value}")

        expost = historical_periods * [mean_value]
        forecast = forecast_periods * [mean_value]
        result_dict = {
            "EXPOST": expost,
            "FORECAST": forecast
        }

        if "ErrorInPeriod" in parameters.keys():
            error_in_periods = [abs(history_value-expost_value)
                               for history_value, expost_value in zip(demand, expost)]
            result_dict.update({"INDEPENDENT_RES01": error_in_periods})
            logger.debug(f"Calculated error in periods: {error_in_periods}")

        logger.info("Successfully completed average calculation")
        return result_dict

    except Exception as e:
        logger.error(f"Error in average_calculation: {str(e)}", exc_info=True)
        raise

def weighted_moving_average_calculation(planning_object_data: Dict, parameters: Dict, historical_periods: int, forecast_periods: int) -> Dict:
    """Calculates Weighted Moving Average Forecast"""
    logger.debug(f"Starting weighted_moving_average_calculation with parameters: {parameters}")
    try:
        demand = planning_object_data["HISTORY"]
        logger.debug(f"Historical demand data: {demand}")

        window = int(parameters["Window"])
        logger.info(f"Using window size: {window}")

        weights = []
        if len(planning_object_data["INDEPENDENT001"]) == historical_periods:
            weights = planning_object_data["INDEPENDENT001"] + \
                [planning_object_data["INDEPENDENT001"][-1]] * forecast_periods
        else:
            weights = planning_object_data["INDEPENDENT001"]

        if "MultiplyWithQuarter" in parameters.keys():
            weights = [w * math.ceil(start_date.month / 3)
                      for w, (start_date, end_date) in zip(weights, planning_object_data["DATETIME"])]
            logger.debug(f"Adjusted weights with quarter multiplier: {weights}")

        weighted_past = [x * w for x, w in zip(demand, weights[:historical_periods])]
        sumed_past_moving_windows = []
        sumed_weight_moving_windows = []

        for index in range(historical_periods - window + 1):
            sumed_past_moving_windows.append(sum(weighted_past[index: index + window]))
            sumed_weight_moving_windows.append(sum(weights[index: index + window]))

        result = [x/w for x, w in zip(sumed_past_moving_windows, sumed_weight_moving_windows)]
        expost = ["NULL"]*window + list(result[:-1])
        forecast = []

        if "Extend" in parameters.keys():
            demand.append(result[-1])
            for i in range(historical_periods, historical_periods + forecast_periods - 1):
                weighted_past.append(demand[-1] * weights[i])
                result.append(sum(weighted_past[-window:]) / sum(weights[i - window + 1: i + 1]))
                demand.append(result[-1])
            forecast = result[-forecast_periods:]
        else:
            forecast = [result[-1]] * forecast_periods

        result_dict = {"EXPOST": expost, "FORECAST": forecast}
        logger.debug(f"Expost values: {expost}")
        logger.debug(f"Forecast values: {forecast}")

        if "ErrorInPeriod" in parameters.keys():
            error_in_periods = ["NULL"] * window + [abs(history_value-expost_value)
                                                   for history_value, expost_value in zip(demand[window:], expost[window:])]
            result_dict.update({"INDEPENDENT_RES01": error_in_periods})
            logger.debug(f"Calculated error in periods: {error_in_periods}")

        logger.info("Successfully completed weighted moving average calculation")
        return result_dict

    except Exception as e:
        logger.error(f"Error in weighted_moving_average_calculation: {str(e)}", exc_info=True)
        raise

def substitute_with_average(data: List) -> List:
    """Substitutes the NULL values with the mean of the non-null values"""
    logger.debug(f"Starting substitute_with_average with data length: {len(data)}")
    try:
        non_null_values = [float(x) for x in filter(lambda x: x != "NULL", data)]
        logger.debug(f"Found {len(non_null_values)} non-null values")

        if len(non_null_values) > 0:
            average_value = sum(non_null_values) / len(non_null_values)
        else:
            average_value = 0.
            logger.warning("No non-null values found, using 0 as default")

        result = [average_value if d == "NULL" else d for d in data]
        logger.debug(f"Result after substitution: {result}")
        return result

    except Exception as e:
        logger.error(f"Error in substitute_with_average: {str(e)}", exc_info=True)
        raise

def substitute_missing_data(planning_object_data: Dict, parameters: Dict) -> Dict:
    """Substitute missing value function"""
    logger.debug(f"Starting substitute_missing_data with correction type: {parameters.get('Correction type')}")
    try:
        result_dict = {}
        if parameters["Correction type"] == "Mean":
            for keyfigure_name, timeseries in planning_object_data.items():
                logger.debug(f"Processing key figure: {keyfigure_name}")
                result = substitute_with_average(timeseries)
                result_dict.update({keyfigure_name: result})

        logger.info("Successfully completed missing data substitution")
        return result_dict

    except Exception as e:
        logger.error(f"Error in substitute_missing_data: {str(e)}", exc_info=True)
        raise

# def evaluate_all_models(X_train, y_train, X_test, y_test, scaler, tscv):
#     """Evaluate all models and return the best performing one with metrics"""
#     models_config = {
#         'Ridge': {'model': Ridge(), 'params': {'alpha': [0.01, 0.1, 1, 10, 100]}},
#         'Lasso': {'model': Lasso(), 'params': {'alpha': [0.0005, 0.001, 0.01, 0.1, 1, 10, 100]}},
#         'ElasticNet': {'model': ElasticNet(max_iter=10000),
#                       'params': {'alpha': [0.001, 0.01, 0.1, 1], 'l1_ratio': [0.1, 0.2, 0.3]}},
#         'RandomForest': {'model': RandomForestRegressor(),
#                         'params': {'n_estimators': [50, 100], 'max_depth': [4, 5], 'max_features': [0.5, 0.6]}},
#         'SVR': {'model': SVR(kernel='linear'), 'params': {'C': [0.1, 1], 'epsilon': [0.05, 0.1]}}
#     }

#     best_model = None
#     best_score = -np.inf
#     best_model_name = ""
#     all_metrics = {}

#     for model_name, model_info in models_config.items():
#         try:
#             logger.info(f"\nEvaluating model: {model_name}")

#             # Grid search with time series cross-validation
#             grid_search = GridSearchCV(
#                 model_info['model'],
#                 model_info['params'],
#                 cv=tscv,
#                 scoring='neg_mean_absolute_error',
#                 n_jobs=-1
#             )
#             grid_search.fit(X_train, y_train)

#             # Get best model from grid search
#             current_model = grid_search.best_estimator_

#             # Evaluate on test data
#             y_pred = current_model.predict(X_test)
#             y_pred_actual = np.expm1(y_pred)
#             y_test_actual = np.expm1(y_test)

#             # Calculate metrics
#             metrics = {
#                 'MAE': mean_absolute_error(y_test_actual, y_pred_actual),
#                 'MAPE': mean_absolute_percentage_error(y_test_actual, y_pred_actual),
#                 'R2': r2_score(y_test_actual, y_pred_actual),
#                 'Best Params': grid_search.best_params_
#             }

#             all_metrics[model_name] = metrics

#             # Print model performance
#             logger.info(f"Model: {model_name}")
#             logger.info(f"Best Parameters: {grid_search.best_params_}")
#             logger.info(f"MAE: {metrics['MAE']:.2f}")
#             logger.info(f"MAPE: {metrics['MAPE']:.2%}")
#             logger.info(f"R2 Score: {metrics['R2']:.2f}")

#             # Update best model if current is better
#             if metrics['R2'] > best_score:
#                 best_score = metrics['R2']
#                 best_model = current_model
#                 best_model_name = model_name

#         except Exception as e:
#             logger.error(f"Error evaluating {model_name}: {str(e)}")
#             continue

#     logger.info(f"\nBest model selected: {best_model_name} with R2 score: {best_score:.2f}")
#     return best_model, best_model_name, all_metrics

def evaluate_all_models(X_train, y_train, X_test, y_test, scaler, tscv, selected_model=None):
    """
    Evaluate all models or a specific model and return the best performing one with metrics
    
    Args:
        selected_model: Optional string to specify which model to use
                       Options: 'Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'SVR',
                               'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost',
                               'KNN', 'ExtraTrees', 'AdaBoost', 'BayesianRidge', 
                               'DecisionTree', 'MLP', 'Auto' (evaluates all)
    """
    models_config = {
        'Ridge': {
            'model': Ridge(),
            'params': {'alpha': [0.01, 0.1, 1, 10, 100]}
        },
        'Lasso': {
            'model': Lasso(),
            'params': {'alpha': [0.0005, 0.001, 0.01, 0.1, 1, 10, 100]}
        },
        'ElasticNet': {
            'model': ElasticNet(max_iter=10000),
            'params': {'alpha': [0.001, 0.01, 0.1, 1], 'l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7]}
        },
        'BayesianRidge': {
            'model': BayesianRidge(),
            'params': {'alpha_1': [1e-6, 1e-5], 'alpha_2': [1e-6, 1e-5]}
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 5, 6],
                'max_features': [0.5, 0.6, 0.7],
                'min_samples_split': [2, 5]
            }
        },
        'ExtraTrees': {
            'model': ExtraTreesRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 5, 6],
                'max_features': [0.5, 0.7]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5],
                'subsample': [0.8, 1.0]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42, verbosity=0),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        },
        'LightGBM': {
            'model': LGBMRegressor(random_state=42, verbosity=-1),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5],
                'num_leaves': [15, 31, 50]
            }
        },
        'CatBoost': {
            'model': CatBoostRegressor(random_state=42, verbose=False),
            'params': {
                'iterations': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'depth': [4, 6, 8]
            }
        },
        'SVR': {
            'model': SVR(kernel='rbf'),
            'params': {
                'C': [0.1, 1, 10],
                'epsilon': [0.01, 0.05, 0.1],
                'gamma': ['scale', 'auto']
            }
        },
        'KNN': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'DecisionTree': {
            'model': DecisionTreeRegressor(random_state=42),
            'params': {
                'max_depth': [3, 4, 5, 6, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'AdaBoost': {
            'model': AdaBoostRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
            }
        },
        'MLP': {
            'model': MLPRegressor(random_state=42, max_iter=2000),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 25)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
    }

    # Determine which models to evaluate
    if selected_model and selected_model != 'Auto':
        if selected_model not in models_config:
            logger.warning(f"Unknown model '{selected_model}'. Available models: {list(models_config.keys())}")
            logger.warning("Evaluating all models instead...")
            models_to_evaluate = models_config
        else:
            models_to_evaluate = {selected_model: models_config[selected_model]}
    else:
        models_to_evaluate = models_config

    best_model = None
    best_score = -np.inf
    best_model_name = ""
    all_metrics = {}

    for model_name, model_info in models_to_evaluate.items():
        try:
            logger.info(f"\nEvaluating model: {model_name}")

            # Grid search with time series cross-validation
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=tscv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            # Get best model from grid search
            current_model = grid_search.best_estimator_

            # Evaluate on test data
            y_pred = current_model.predict(X_test)
            y_pred_actual = np.expm1(y_pred)
            y_test_actual = np.expm1(y_test)

            # Calculate metrics
            metrics = {
                'MAE': mean_absolute_error(y_test_actual, y_pred_actual),
                'MAPE': mean_absolute_percentage_error(y_test_actual, y_pred_actual),
                'R2': r2_score(y_test_actual, y_pred_actual),
                'Best Params': grid_search.best_params_
            }

            all_metrics[model_name] = metrics

            # Print model performance
            logger.info(f"Model: {model_name}")
            logger.info(f"Best Parameters: {grid_search.best_params_}")
            logger.info(f"MAE: {metrics['MAE']:.2f}")
            logger.info(f"MAPE: {metrics['MAPE']:.2%}")
            logger.info(f"R2 Score: {metrics['R2']:.2f}")

            # Update best model if current is better
            if metrics['R2'] > best_score:
                best_score = metrics['R2']
                best_model = current_model
                best_model_name = model_name

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            continue

    if best_model is None:
        logger.error("No models were successfully evaluated")
        raise ValueError("Model evaluation failed for all models")

    logger.info(f"\nBest model selected: {best_model_name} with R2 score: {best_score:.2f}")
    
    # Log comparison of all evaluated models
    if len(all_metrics) > 1:
        logger.info("\n=== Model Comparison Summary ===")
        sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]['R2'], reverse=True)
        for rank, (name, metrics) in enumerate(sorted_models, 1):
            logger.info(f"{rank}. {name}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, MAPE={metrics['MAPE']:.2%}")
    
    return best_model, best_model_name, all_metrics

# def ml_forecast_calculation(planning_object_data: Dict, parameters: Dict, historical_periods: int, forecast_periods: int, date_list: List[Tuple]) -> Dict:
#     """Calculates ML-based Forecast with proper period handling"""
#     logger.info(f"Starting ML forecast calculation with model: {parameters.get('Model')}")

#     try:
#         # Minimum data check
#         if historical_periods < 6:
#             logger.warning(f"Insufficient history ({historical_periods} periods). Returning NULL forecast")
#             return {
#                 "EXPOST": ["NULL"] * historical_periods,
#                 "FORECAST": ["NULL"] * forecast_periods
#             }

#         # Store original demand data for expost
#         original_demand = planning_object_data["HISTORY"].copy()
#         logger.info(f"\nOriginal Input Data (first 10 values): {original_demand[:10]}")

#         demand_history = planning_object_data["HISTORY"]

#         # Ensure we have exactly the right number of historical periods
#         if len(demand_history) != historical_periods:
#             logger.warning(f"Demand history length ({len(demand_history)}) doesn't match historical_periods ({historical_periods})")
#             if len(demand_history) < historical_periods:
#                 demand_history.extend([0.0] * (historical_periods - len(demand_history)))
#             else:
#                 demand_history = demand_history[:historical_periods]

#         # Create DataFrame with all historical data
#         history_dates = [d[0] for d in date_list[:historical_periods]]
#         df_history = pd.DataFrame({
#             'Date': history_dates,
#             'Actuals Qty for sales (S4+ETO)': demand_history,
#             'original_index': list(range(historical_periods))
#         })

#         # Data preprocessing
#         df_history['Date'] = pd.to_datetime(df_history['Date'])
#         df_history = df_history.sort_values(by='Date')

#         # Identify valid values for modeling
#         valid_mask = (df_history['Actuals Qty for sales (S4+ETO)'].notna() &
#                      np.isfinite(df_history['Actuals Qty for sales (S4+ETO)']) &
#                      (df_history['Actuals Qty for sales (S4+ETO)'] != 0))

#         df_valid = df_history[valid_mask].copy()

#         if len(df_valid) < 6:
#             logger.warning(f"Insufficient valid data points ({len(df_valid)}) for modeling")
#             return {
#                 "EXPOST": ["NULL"] * historical_periods,
#                 "FORECAST": ["NULL"] * forecast_periods
#             }

#         # Outlier handling
#         Q1 = df_valid["Actuals Qty for sales (S4+ETO)"].quantile(0.25)
#         Q3 = df_valid["Actuals Qty for sales (S4+ETO)"].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR

#         df_valid["Sales_clean"] = df_valid["Actuals Qty for sales (S4+ETO)"].clip(lower_bound, upper_bound)
#         df_valid['log_sales'] = np.log1p(df_valid["Sales_clean"])

#         # Feature engineering
#         df_valid['Year'] = df_valid['Date'].dt.year
#         df_valid['Month'] = df_valid['Date'].dt.month
#         df_valid['Quarter'] = df_valid['Date'].dt.quarter
#         df_valid['Is_High_Season'] = df_valid['Month'].isin([12, 1, 2]).astype(int)
#         df_valid['Lag_1'] = df_valid['log_sales'].shift(1)
#         df_valid['Lag_2'] = df_valid['log_sales'].shift(2)
#         df_valid['Lag_3'] = df_valid['log_sales'].shift(3)
#         df_valid['Rolling_3'] = df_valid['log_sales'].rolling(window=3).mean()

#         # Drop rows with NaN in features
#         df_model = df_valid.dropna(subset=['Lag_1', 'Lag_2', 'Lag_3', 'Rolling_3']).copy()

#         if len(df_model) < 3:
#             logger.warning("Insufficient data after feature engineering")
#             return {
#                 "EXPOST": ["NULL"] * historical_periods,
#                 "FORECAST": ["NULL"] * forecast_periods
#             }

#         # Prepare features and target
#         features = ['Year', 'Month', 'Quarter', 'Lag_1', 'Lag_2', 'Lag_3', 'Rolling_3', 'Is_High_Season']
#         target = 'log_sales'

#         # Split data into train and test sets (80-20 split)
#         split_idx = int(len(df_model) * 0.8)
#         X_train = df_model[features].iloc[:split_idx]
#         y_train = df_model[target].iloc[:split_idx]
#         X_test = df_model[features].iloc[split_idx:]
#         y_test = df_model[target].iloc[split_idx:]

#         # Scale features
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)

#         # Time series cross-validation
#         tscv = TimeSeriesSplit(n_splits=min(3, len(X_train) - 1))

#         # Evaluate all models and select the best one
#         best_model, best_model_name, all_metrics = evaluate_all_models(
#             X_train_scaled, y_train, X_test_scaled, y_test, scaler, tscv
#         )

#         # Train final model on all available data
#         X_full = df_model[features]
#         y_full = df_model[target]
#         X_full_scaled = scaler.fit_transform(X_full)
#         best_model.fit(X_full_scaled, y_full)

#         # Generate predictions for the modeling data
#         model_predictions_log = best_model.predict(X_full_scaled)
#         model_predictions = np.expm1(model_predictions_log)

#         # Create expost array with original values (not the preprocessed ones)
#         expost = original_demand.copy()

#         # Only replace values where we have predictions with predicted values
#         for i, (idx, pred) in enumerate(zip(df_model['original_index'].values, model_predictions)):
#             if 0 <= idx < historical_periods:
#                 expost[idx] = float(pred)

#         # Forecasting future periods
#         if len(df_valid) >= 3:
#             initial_lags_for_forecast = df_valid['log_sales'].iloc[-3:].values.tolist()
#         else:
#             available_logs = df_valid['log_sales'].dropna().values
#             if len(available_logs) > 0:
#                 initial_lags_for_forecast = available_logs[-min(3, len(available_logs)):].tolist()
#                 while len(initial_lags_for_forecast) < 3:
#                     initial_lags_for_forecast.insert(0, initial_lags_for_forecast[0] if initial_lags_for_forecast else 0.0)
#             else:
#                 initial_lags_for_forecast = [0.0, 0.0, 0.0]

#         future_dates = [d[0] for d in date_list[historical_periods:historical_periods + forecast_periods]]

#         def recursive_forecast(model, initial_lags, future_dates, scaler):
#             lags = list(initial_lags)
#             predictions = []
#             for i, current_date in enumerate(future_dates):
#                 year = current_date.year
#                 month = current_date.month
#                 quarter = (month - 1) // 3 + 1
#                 is_high_season = 1 if month in [12, 1, 2] else 0
#                 rolling_3 = np.mean(lags[-min(3, len(lags)):]) if len(lags) > 0 else 0
#                 lag_1 = lags[-1] if len(lags) >= 1 else 0
#                 lag_2 = lags[-2] if len(lags) >= 2 else 0
#                 lag_3 = lags[-3] if len(lags) >= 3 else 0

#                 X_future = np.array([year, month, quarter, lag_1, lag_2, lag_3,
#                                     rolling_3, is_high_season]).reshape(1, -1)
#                 X_future_scaled = scaler.transform(X_future)
#                 pred_log = model.predict(X_future_scaled)[0]
#                 predictions.append(np.expm1(pred_log))
#                 lags.append(pred_log)
#                 if len(lags) > 3:
#                     lags = lags[-3:]

#             return predictions

#         forecast = recursive_forecast(best_model, initial_lags_for_forecast, future_dates, scaler)

#         logger.info(f"\nFinal Forecast Values: {forecast}")

#         # Ensure exact lengths
#         if len(expost) != historical_periods:
#             expost = (expost[:historical_periods] if len(expost) > historical_periods
#                      else expost + ["NULL"] * (historical_periods - len(expost)))

#         if len(forecast) != forecast_periods:
#             forecast = (forecast[:forecast_periods] if len(forecast) > forecast_periods
#                        else forecast + [forecast[-1] if forecast else 0.0] * (forecast_periods - len(forecast)))

#         # Clean and validate results
#         expost_clean = validate_and_clean(expost)
#         forecast_clean = validate_and_clean(forecast)

#         result_dict = {"EXPOST": expost_clean, "FORECAST": forecast_clean}

#         if "ErrorInPeriod" in parameters.keys():
#             error_in_periods = []
#             for i in range(historical_periods):
#                 if (i < len(original_demand) and
#                     expost_clean[i] != "NULL" and
#                     original_demand[i] is not None and
#                     np.isfinite(original_demand[i])):
#                     error_in_periods.append(abs(original_demand[i] - float(expost_clean[i])))
#                 else:
#                     error_in_periods.append("NULL")
#             result_dict.update({"INDEPENDENT_RES01": error_in_periods})

#         logger.info("Successfully completed ML forecast calculation")
#         return result_dict

#     except Exception as e:
#         logger.error(f"Error in ml_forecast_calculation: {str(e)}", exc_info=True)
#         return {"EXPOST": ["NULL"] * historical_periods, "FORECAST": ["NULL"] * forecast_periods}

def ml_forecast_calculation(planning_object_data: Dict, parameters: Dict, 
                           historical_periods: int, forecast_periods: int, 
                           date_list: List[Tuple]) -> Dict:
    """
    Calculates ML-based Forecast with proper period handling
    
    Parameters can include:
    - Model: 'Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'SVR', 'GradientBoosting',
             'XGBoost', 'LightGBM', 'CatBoost', 'KNN', 'ExtraTrees', 'AdaBoost',
             'BayesianRidge', 'DecisionTree', 'MLP', or 'Auto' (default)
    """
    selected_model = parameters.get('Model', 'Auto')
    logger.info(f"Starting ML forecast calculation with model: {selected_model}")

    try:
        # Minimum data check
        if historical_periods < 6:
            logger.warning(f"Insufficient history ({historical_periods} periods). Returning NULL forecast")
            return {
                "EXPOST": ["NULL"] * historical_periods,
                "FORECAST": ["NULL"] * forecast_periods
            }

        # Store original demand data for expost
        original_demand = planning_object_data["HISTORY"].copy()
        logger.info(f"\nOriginal Input Data (first 10 values): {original_demand[:10]}")

        demand_history = planning_object_data["HISTORY"]

        # Ensure we have exactly the right number of historical periods
        if len(demand_history) != historical_periods:
            logger.warning(f"Demand history length ({len(demand_history)}) doesn't match historical_periods ({historical_periods})")
            if len(demand_history) < historical_periods:
                demand_history.extend([0.0] * (historical_periods - len(demand_history)))
            else:
                demand_history = demand_history[:historical_periods]

        # Create DataFrame with all historical data
        history_dates = [d[0] for d in date_list[:historical_periods]]
        df_history = pd.DataFrame({
            'Date': history_dates,
            'Actuals Qty for sales (S4+ETO)': demand_history,
            'original_index': list(range(historical_periods))
        })

        # Data preprocessing
        df_history['Date'] = pd.to_datetime(df_history['Date'])
        df_history = df_history.sort_values(by='Date')

        # Identify valid values for modeling
        valid_mask = (df_history['Actuals Qty for sales (S4+ETO)'].notna() &
                     np.isfinite(df_history['Actuals Qty for sales (S4+ETO)']) &
                     (df_history['Actuals Qty for sales (S4+ETO)'] != 0))

        df_valid = df_history[valid_mask].copy()

        if len(df_valid) < 6:
            logger.warning(f"Insufficient valid data points ({len(df_valid)}) for modeling")
            return {
                "EXPOST": ["NULL"] * historical_periods,
                "FORECAST": ["NULL"] * forecast_periods
            }

        # Outlier handling
        Q1 = df_valid["Actuals Qty for sales (S4+ETO)"].quantile(0.25)
        Q3 = df_valid["Actuals Qty for sales (S4+ETO)"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_valid["Sales_clean"] = df_valid["Actuals Qty for sales (S4+ETO)"].clip(lower_bound, upper_bound)
        df_valid['log_sales'] = np.log1p(df_valid["Sales_clean"])

        # Feature engineering
        df_valid['Year'] = df_valid['Date'].dt.year
        df_valid['Month'] = df_valid['Date'].dt.month
        df_valid['Quarter'] = df_valid['Date'].dt.quarter
        df_valid['Is_High_Season'] = df_valid['Month'].isin([12, 1, 2]).astype(int)
        df_valid['Lag_1'] = df_valid['log_sales'].shift(1)
        df_valid['Lag_2'] = df_valid['log_sales'].shift(2)
        df_valid['Lag_3'] = df_valid['log_sales'].shift(3)
        df_valid['Rolling_3'] = df_valid['log_sales'].rolling(window=3).mean()
        df_valid['Rolling_6'] = df_valid['log_sales'].rolling(window=6).mean()
        df_valid['Trend'] = np.arange(len(df_valid))

        # Drop rows with NaN in features
        df_model = df_valid.dropna(subset=['Lag_1', 'Lag_2', 'Lag_3', 'Rolling_3']).copy()

        if len(df_model) < 3:
            logger.warning("Insufficient data after feature engineering")
            return {
                "EXPOST": ["NULL"] * historical_periods,
                "FORECAST": ["NULL"] * forecast_periods
            }

        # Prepare features and target
        features = ['Year', 'Month', 'Quarter', 'Lag_1', 'Lag_2', 'Lag_3', 
                   'Rolling_3', 'Is_High_Season', 'Trend']
        
        # Add Rolling_6 if available
        if 'Rolling_6' in df_model.columns and df_model['Rolling_6'].notna().sum() > 0:
            features.append('Rolling_6')
            df_model = df_model.dropna(subset=['Rolling_6'])
        
        target = 'log_sales'

        # Split data into train and test sets (80-20 split)
        split_idx = int(len(df_model) * 0.8)
        X_train = df_model[features].iloc[:split_idx]
        y_train = df_model[target].iloc[:split_idx]
        X_test = df_model[features].iloc[split_idx:]
        y_test = df_model[target].iloc[split_idx:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=min(3, len(X_train) - 1))

        # Evaluate models and select the best one
        best_model, best_model_name, all_metrics = evaluate_all_models(
            X_train_scaled, y_train, X_test_scaled, y_test, scaler, tscv, selected_model
        )

        # Train final model on all available data
        X_full = df_model[features]
        y_full = df_model[target]
        X_full_scaled = scaler.fit_transform(X_full)
        best_model.fit(X_full_scaled, y_full)

        # Generate predictions for the modeling data
        model_predictions_log = best_model.predict(X_full_scaled)
        model_predictions = np.expm1(model_predictions_log)

        # Create expost array with original values
        expost = original_demand.copy()

        # Replace values where we have predictions
        for i, (idx, pred) in enumerate(zip(df_model['original_index'].values, model_predictions)):
            if 0 <= idx < historical_periods:
                expost[idx] = float(pred)

        # Forecasting future periods
        if len(df_valid) >= 3:
            initial_lags_for_forecast = df_valid['log_sales'].iloc[-3:].values.tolist()
        else:
            available_logs = df_valid['log_sales'].dropna().values
            if len(available_logs) > 0:
                initial_lags_for_forecast = available_logs[-min(3, len(available_logs)):].tolist()
                while len(initial_lags_for_forecast) < 3:
                    initial_lags_for_forecast.insert(0, initial_lags_for_forecast[0] if initial_lags_for_forecast else 0.0)
            else:
                initial_lags_for_forecast = [0.0, 0.0, 0.0]

        future_dates = [d[0] for d in date_list[historical_periods:historical_periods + forecast_periods]]

        def recursive_forecast(model, initial_lags, future_dates, scaler, features):
            lags = list(initial_lags)
            predictions = []
            trend_value = len(df_model)  # Continue trend from last training point
            
            for i, current_date in enumerate(future_dates):
                year = current_date.year
                month = current_date.month
                quarter = (month - 1) // 3 + 1
                is_high_season = 1 if month in [12, 1, 2] else 0
                rolling_3 = np.mean(lags[-min(3, len(lags)):]) if len(lags) > 0 else 0
                lag_1 = lags[-1] if len(lags) >= 1 else 0
                lag_2 = lags[-2] if len(lags) >= 2 else 0
                lag_3 = lags[-3] if len(lags) >= 3 else 0

                # Build feature vector based on what was used in training
                feature_values = [year, month, quarter, lag_1, lag_2, lag_3, 
                                rolling_3, is_high_season, trend_value]
                
                # Add rolling_6 if it was used in training
                if 'Rolling_6' in features:
                    rolling_6 = np.mean(lags[-min(6, len(lags)):]) if len(lags) > 0 else 0
                    feature_values.append(rolling_6)
                
                X_future = np.array(feature_values).reshape(1, -1)
                X_future_scaled = scaler.transform(X_future)
                pred_log = model.predict(X_future_scaled)[0]
                predictions.append(np.expm1(pred_log))
                lags.append(pred_log)
                if len(lags) > 6:
                    lags = lags[-6:]
                trend_value += 1

            return predictions

        forecast = recursive_forecast(best_model, initial_lags_for_forecast, future_dates, scaler, features)

        logger.info(f"\nFinal Forecast Values: {forecast}")

        # Ensure exact lengths
        if len(expost) != historical_periods:
            expost = (expost[:historical_periods] if len(expost) > historical_periods
                     else expost + ["NULL"] * (historical_periods - len(expost)))

        if len(forecast) != forecast_periods:
            forecast = (forecast[:forecast_periods] if len(forecast) > forecast_periods
                       else forecast + [forecast[-1] if forecast else 0.0] * (forecast_periods - len(forecast)))

        # Validate results
        from typing import List, Union
        
        def validate_and_clean(values: List[Union[float, str]]) -> List[Union[float, str]]:
            cleaned = []
            for i, v in enumerate(values):
                if isinstance(v, (float, int)) and not np.isfinite(v):
                    logger.warning(f"Invalid value detected at position {i}: {v}. Replacing with 'NULL'")
                    cleaned.append("NULL")
                else:
                    cleaned.append(v)
            return cleaned
        
        expost_clean = validate_and_clean(expost)
        forecast_clean = validate_and_clean(forecast)

        result_dict = {"EXPOST": expost_clean, "FORECAST": forecast_clean}

        if "ErrorInPeriod" in parameters.keys():
            error_in_periods = []
            for i in range(historical_periods):
                if (i < len(original_demand) and
                    expost_clean[i] != "NULL" and
                    original_demand[i] is not None and
                    np.isfinite(original_demand[i])):
                    error_in_periods.append(abs(original_demand[i] - float(expost_clean[i])))
                else:
                    error_in_periods.append("NULL")
            result_dict.update({"INDEPENDENT_RES01": error_in_periods})

        logger.info(f"Successfully completed ML forecast calculation using {best_model_name}")
        return result_dict

    except Exception as e:
        logger.error(f"Error in ml_forecast_calculation: {str(e)}", exc_info=True)
        return {"EXPOST": ["NULL"] * historical_periods, "FORECAST": ["NULL"] * forecast_periods}

def calculate_forecast(planning_object: Dict, alogrithm_name: str, parameters: Dict,
                      historical_periods: int, forecast_periods: int, date_list: List[Tuple]) -> Dict:
    """Forecast calculation function"""
    logger.info(f"Starting calculate_forecast with algorithm: {alogrithm_name}")
    try:
        planning_object_data = {}
        logger.debug(f"Processing planning object with GroupID: {planning_object.get('GroupID')}")

        for data in planning_object["_AlgorithmDataInput"]:
            if data["SemanticKeyFigure"] == "HISTORY":
                # Convert and clean history data
                history_str_list = data["TimeSeries"].split(';')[:historical_periods]
                history_float_list = safe_float_convert(history_str_list)
                planning_object_data.update({"HISTORY": history_float_list})

                start_idx = data["FirstPeriodIndex"] - 1
                end_idx = start_idx + historical_periods + forecast_periods
                planning_object_data.update({"DATETIME": date_list[start_idx:end_idx]})
                logger.debug(f"Processed HISTORY data with {len(planning_object_data['HISTORY'])} periods")

            elif "KEYFIGURE_IN" in data["SemanticKeyFigure"]:
                # Keep as string for keyfigure inputs
                planning_object_data.update(
                    {data["SemanticKeyFigure"]: [x for x in data["TimeSeries"].split(';')[:historical_periods]]})
            else:
                # Convert other numerical key figures
                str_list = data["TimeSeries"].split(';')[:historical_periods]
                planning_object_data.update(
                    {data["SemanticKeyFigure"]: safe_float_convert(str_list)})

        results = {}
        if alogrithm_name == "Average":
            logger.info("Executing Average algorithm")
            results = average_calculation(
                planning_object_data, parameters, historical_periods, forecast_periods)
        elif alogrithm_name == "Weighted MA":
            logger.info("Executing Weighted MA algorithm")
            results = weighted_moving_average_calculation(
                planning_object_data, parameters, historical_periods, forecast_periods)
        elif alogrithm_name == "SubstMissing":
            logger.info("Executing SubstMissing algorithm")
            results = substitute_missing_data(
                planning_object_data, parameters)
        elif alogrithm_name == "MLForecast":
            logger.info("Executing MLForecast algorithm")
            results = ml_forecast_calculation(
                planning_object_data, parameters, historical_periods, forecast_periods, date_list)
        else:
            logger.warning(f"Unknown algorithm name: {alogrithm_name}")

        # Final validation of all results
        for kf_name, kf_values in results.items():
            if any(isinstance(v, float) and not np.isfinite(v) for v in kf_values):
                logger.error(f"Non-finite values found in {kf_name} results")
                results[kf_name] = validate_and_clean(kf_values)

        logger.info(f"Completed calculate_forecast for algorithm: {alogrithm_name}")
        return results

    except Exception as e:
        logger.error(f"Error in calculate_forecast: {str(e)}", exc_info=True)
        return {}