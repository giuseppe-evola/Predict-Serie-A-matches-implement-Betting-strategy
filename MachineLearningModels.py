import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import uniform, randint
import os
import pickle
import warnings
warnings.filterwarnings('ignore')


# To save the fitted models in the directory ML models

def save_model(model, filename):
    os.makedirs("ML_models", exist_ok=True)  # Crea la directory se non esiste
    filepath = os.path.join("ML_models", filename)
    with open(filepath, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to: {filepath}")





# Logistic Regression (Simple)
def train_simple_logistic(X, y, X_test, y_test):
    print("\nSimple Logistic Regression:")
    
    param_dist = {
        'C': uniform(0.001, 10.0),
        'solver': ['lbfgs', 'newton-cg', 'sag'],
        'max_iter': [1000]
    }
    
    model = LogisticRegression(multi_class='multinomial', random_state=42)
    
    search = RandomizedSearchCV(
        model, 
        param_distributions=param_dist,
        n_iter=20,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=["accuracy", "f1_weighted"],
        refit='f1_weighted',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    search.fit(X, y)
    
    print("\nBest parameters found:")
    print(search.best_params_)
    print("\nBest cross-validation scores:")
    for metric in search.scoring:
        print(f"{metric}: {search.cv_results_[f'mean_test_{metric}'][search.best_index_]:.3f}")
    
    y_pred = search.predict(X_test)
    print("\nClassification Report on test set:")
    print(classification_report(y_test, y_pred))
    
    save_model(search.best_estimator_, "logistic_regression_simple.pkl")
    print("------------------------------------------------------------------------------------------")
    return search.best_estimator_

# Elastic Net Logistic Regression
def train_elastic_net(X, y, X_test, y_test):
    print("\nElastic Net Logistic Regression:")
    
    param_dist = {
        'C': uniform(0.001, 10.0),
        'l1_ratio': uniform(0, 1),
        'solver': ['saga'],
        'penalty': ['elasticnet'],
        'max_iter': [2000]
    }
    
    model = LogisticRegression(multi_class='multinomial', random_state=42)
    
    search = RandomizedSearchCV(
        model, 
        param_distributions=param_dist,
        n_iter=50,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=["accuracy", "f1_weighted"],
        refit='f1_weighted',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    search.fit(X, y)
    
    print("\nBest parameters found:")
    print(search.best_params_)
    print("\nBest cross-validation scores:")
    for metric in search.scoring:
        print(f"{metric}: {search.cv_results_[f'mean_test_{metric}'][search.best_index_]:.3f}")
    
    y_pred = search.predict(X_test)
    print("\nClassification Report on test set:")
    print(classification_report(y_test, y_pred))

    try:
        save_model(search.best_estimator_, "logistic_regression_elastic_net.pkl")
    except Exception as e:
        print(f"Error during model saving: {e}")
    print("------------------------------------------------------------------------------------------")
    return search.best_estimator_



# Random Forest with two-phase tuning
def train_random_forest_improved(X, y, X_test, y_test):
    print("\nRandom Forest Improved:")
    
    param_dist = {                                 # Phase 1 hyperparateter tuning (broader search) 
        'n_estimators': randint(100, 500),
        'max_depth': randint(10, 50),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': uniform(0.1, 0.9),
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    model = RandomForestClassifier(random_state=42) # model initialization
    
    search1 = RandomizedSearchCV(        
        model,
        param_distributions=param_dist,
        n_iter=30,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_weighted',  # metric to score outcome of the mdoel with different parameters
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    
    search1.fit(X, y)  # Best version of the model after the 1 phase of hyoerparameter tuning                       
    best_params = search1.best_params_

    refined_param_dist = {    # Phase 2 hyperparateter tuning (deeper search) 
        'n_estimators': randint(max(100, best_params['n_estimators'] - 50), min(500, best_params['n_estimators'] + 50)),
        'max_depth': randint(max(10, best_params['max_depth'] - 5), min(50, best_params['max_depth'] + 5)),
        'min_samples_split': randint(max(2, best_params['min_samples_split'] - 2), min(20, best_params['min_samples_split'] + 2)),
        'min_samples_leaf': randint(max(1, best_params['min_samples_leaf'] - 1), min(10, best_params['min_samples_leaf'] + 1)),
        'max_features': uniform(max(0.1, float(best_params['max_features'] - 0.1)), min(0.9, float(best_params['max_features'] + 0.1))),
        'bootstrap': [best_params['bootstrap']],
        'class_weight': [best_params['class_weight']]
    }
    
    search2 = RandomizedSearchCV(
        model,
        param_distributions=refined_param_dist,
        n_iter=20,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_weighted',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    search2.fit(X, y)   # best model after 2 phase hyperparameter tuning
    
    print("\nBest final parameters:")
    print(search2.best_params_)
    print(f"\nBest f1_weighted: {search2.best_score_:.3f}")
    
    y_pred = search2.predict(X_test)
    print("\nClassification Report on test set:")
    print(classification_report(y_test, y_pred))


    try:
        save_model(search2.best_estimator_, "random_forest.pkl")
    except Exception as e:
        print(f"Error during model saving: {e}")
    print("------------------------------------------------------------------------------------------")
    return search2.best_estimator_

# XGBoost with two-phase tuning
def train_xgboost_improved(X, y, X_test, y_test):
    print("\nXGBoost Improved:")
    
    param_dist = {
        # Phase 1 hyperparateter tuning (broader search) 
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 7),
        'gamma': uniform(0, 0.5),
        'scale_pos_weight': uniform(0.8, 0.4)
    }
    
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    
    search1 = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=30,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_weighted',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    search1.fit(X, y) # best model after 1 phase hyperparameter tuning
    best_params = search1.best_params_
    
    refined_param_dist = {
        # Phase 2 hyperparateter tuning (deeper search) 
        'n_estimators': randint(max(100, best_params['n_estimators'] - 50), min(500, best_params['n_estimators'] + 50)),
        'max_depth': randint(max(3, best_params['max_depth'] - 2), min(10, best_params['max_depth'] + 2)),
        'learning_rate': uniform(max(0.01, best_params['learning_rate'] - 0.05), min(0.3, best_params['learning_rate'] + 0.05)),
        'subsample': uniform(max(0.6, best_params['subsample'] - 0.1), min(1.0, best_params['subsample'] + 0.1)),
        'colsample_bytree': uniform(max(0.6, best_params['colsample_bytree'] - 0.1), min(1.0, best_params['colsample_bytree'] + 0.1)),
        'min_child_weight': randint(max(1, best_params['min_child_weight'] - 1), min(7, best_params['min_child_weight'] + 1)),
        'gamma': uniform(max(0, best_params['gamma'] - 0.1), min(0.5, best_params['gamma'] + 0.1)),
        'scale_pos_weight': uniform(max(0.8, best_params['scale_pos_weight'] - 0.1), min(1.2, best_params['scale_pos_weight'] + 0.1))
    }
    
    search2 = RandomizedSearchCV(
        model,
        param_distributions=refined_param_dist,
        n_iter=20,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_weighted',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    search2.fit(X, y) # best model after 2 phase hyperparameter tuning
    
    print("\nBest final parameters:")
    print(search2.best_params_)
    print(f"\nBest f1_weighted-score: {search2.best_score_:.3f}")
    
    y_pred = search2.predict(X_test)
    print("\nClassification Report on test set:")
    print(classification_report(y_test, y_pred))

    try:
        save_model(search2.best_estimator_, "xgboost_classifier.pkl")
    except Exception as e:
        print(f"Error during model saving: {e}")
    print("------------------------------------------------------------------------------------------")
    return search2.best_estimator_

# LightGBM with two-phase tuning
def train_lightgbm_improved(X, y, X_test, y_test):
    print("\nLightGBM Improved:")
    
    param_dist = {
        # Phase 1 hyperparateter tuning (broader search) 
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_samples': randint(10, 50),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)
    }
    
    model = LGBMClassifier(random_state=42, verbosity=-1)
    
    search1 = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=30,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_weighted',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    search1.fit(X, y) # best model after 1 phase hyperparameter tuning
    best_params = search1.best_params_
    
    refined_param_dist = {
        # Phase 2 hyperparateter tuning (broader search) 
        'n_estimators': randint(max(100, best_params['n_estimators'] - 50), min(500, best_params['n_estimators'] + 50)),
        'max_depth': randint(max(3, best_params['max_depth'] - 2), min(10, best_params['max_depth'] + 2)),
        'learning_rate': uniform(max(0.01, best_params['learning_rate'] - 0.05), min(0.3, best_params['learning_rate'] + 0.05)),
        'subsample': uniform(max(0.6, best_params['subsample'] - 0.1), min(1.0, best_params['subsample'] + 0.1)),
        'colsample_bytree': uniform(max(0.6, best_params['colsample_bytree'] - 0.1), min(1.0, best_params['colsample_bytree'] + 0.1)),
        'min_child_samples': randint(max(10, best_params['min_child_samples'] - 5), min(50, best_params['min_child_samples'] + 5)),
        'reg_alpha': uniform(max(0, best_params['reg_alpha'] - 0.1), min(1, best_params['reg_alpha'] + 0.1)),
        'reg_lambda': uniform(max(0, best_params['reg_lambda'] - 0.1), min(1, best_params['reg_lambda'] + 0.1))
    }
    
    search2 = RandomizedSearchCV(
        model,
        param_distributions=refined_param_dist,
        n_iter=20,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_weighted',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    search2.fit(X, y) # best model after 2 phase hyperparameter tuning
    
    print("\nBest final parameters:")
    print(search2.best_params_)
    print(f"\nBest f1_weighted-score: {search2.best_score_:.3f}")
    
    y_pred = search2.predict(X_test)
    print("\nClassification Report on test set:")
    print(classification_report(y_test, y_pred))

    try:
        save_model(search2.best_estimator_, "lightgbm_improved.pkl")
    except Exception as e:
        print(f"Error during model saving: {e}")
    print("------------------------------------------------------------------------------------------")
    return search2.best_estimator_

# Comparative Evaluation
def evaluate_models(models, X_test, y_test):
    print("\nComparative Model Evaluation:")
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = report['weighted avg']['f1-score']
        print(f"\n{name}:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    # Select the best model based on the highest weighted F1-score
    best_model_name = max(results, key=results.get)
    print(f"\nThe best model is: {best_model_name} with a weighted F1-score of {results[best_model_name]:.3f}")
    
    return best_model_name, models[best_model_name]

