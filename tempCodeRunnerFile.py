# Product Demand Forecasting Mini Project
# Author: Nishant
# Description: Complete machine learning pipeline for product demand forecasting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random state for reproducibility
RANDOM_STATE = 42

class ProductDemandForecasting:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self, filepath):
        """Load and display basic information about the dataset"""
        print("=" * 60)
        print("STEP 1: LOADING DATASET")
        print("=" * 60)
        
        try:
            self.data = pd.read_csv(filepath)
            print(f"Dataset loaded successfully!")
            print(f"Dataset shape: {self.data.shape}")
            print(f"\nColumn names: {list(self.data.columns)}")
            print(f"\nFirst 5 rows:")
            print(self.data.head())
            return True
        except FileNotFoundError:
            print("Dataset file not found. Please ensure you've downloaded it from:")
            print("https://www.kaggle.com/felixzhao/productdemandforecasting")
            return False
    
    def explore_data(self):
        """Explore the dataset and handle missing values"""
        print("\n" + "=" * 60)
        print("STEP 2: DATA EXPLORATION & CLEANING")
        print("=" * 60)
        
        # Check data types
        print("Data types:")
        print(self.data.dtypes)
        
        # Check for missing values
        print(f"\nMissing values per column:")
        missing_values = self.data.isnull().sum()
        print(missing_values)
        
        if missing_values.sum() > 0:
            print(f"Dropping {missing_values.sum()} rows with missing values...")
            self.data = self.data.dropna()
            print(f"New dataset shape: {self.data.shape}")
        else:
            print("No missing values found!")
        
        # Display basic statistics
        print(f"\nDataset statistics:")
        print(self.data.describe(include='all'))
        
        # --- NEW CODE TO FIX THE VALUEERROR ---
        print("\nCleaning 'Order_Demand' column...")
        self.data['Order_Demand'] = self.data['Order_Demand'].astype(str).str.replace('(', '').str.replace(')', '')
        self.data['Order_Demand'] = pd.to_numeric(self.data['Order_Demand'], errors='coerce')
        self.data = self.data.dropna(subset=['Order_Demand'])
        print(f"Cleaned dataset shape: {self.data.shape}")
        # --- END OF NEW CODE ---
        
    def feature_selection(self):
        """Select and prepare features for modeling"""
        print("\n" + "=" * 60)
        print("STEP 3: FEATURE SELECTION & PREPROCESSING")
        print("=" * 60)
        
        # Display all available columns
        print("Available columns:", list(self.data.columns))
        
        # Assume the target variable is related to demand/order quantity
        target_candidates = ['Order_Demand', 'demand', 'quantity', 'sales', 'target']
        target_column = None
        
        for col in target_candidates:
            if col in self.data.columns:
                target_column = col
                break
        
        if target_column is None:
            target_column = self.data.columns[-1]
            print(f"WARNING: Using '{target_column}' as target variable")
        else:
            print(f"Found target variable: '{target_column}'")
        
        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            print(f"Encoding categorical variables: {list(categorical_columns)}")
            le = LabelEncoder()
            for col in categorical_columns:
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle date columns if any
        date_columns = X.select_dtypes(include=['datetime64']).columns
        if len(date_columns) > 0:
            for col in date_columns:
                X[f'{col}_year'] = pd.to_datetime(X[col]).dt.year
                X[f'{col}_month'] = pd.to_datetime(X[col]).dt.month
                X[f'{col}_day'] = pd.to_datetime(X[col]).dt.day
                X = X.drop(columns=[col])
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        print(f"Selected features: {list(X.columns)}")
        
        return X, y
    
    def split_data(self, X, y):
        """Split data into train, validation, and test sets"""
        print("\n" + "=" * 60)
        print("STEP 4: DATA SPLITTING")
        print("=" * 60)
        
        # First split: 80% train, 20% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        
        # Second split: 10% validation, 10% test from the 20% temp
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
        )
        
        print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_features(self, X_train, X_val, X_test):
        """Scale features using StandardScaler"""
        print("\n" + "=" * 60)
        print("STEP 5: FEATURE SCALING")
        print("=" * 60)
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Features scaled using StandardScaler")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def train_decision_tree(self, X_train, X_val, y_train, y_val):
        """Train Decision Tree with different criteria"""
        print("\n" + "=" * 60)
        print("STEP 6: DECISION TREE REGRESSION")
        print("=" * 60)
        
        criteria = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
        best_criterion = None
        best_score = -np.inf
        dt_results = {}
        
        print("Testing different criteria for Decision Tree:")
        
        for criterion in criteria:
            try:
                # Train model
                dt = DecisionTreeRegressor(
                    criterion=criterion,
                    random_state=RANDOM_STATE,
                    max_depth=10,  # Prevent overfitting
                    min_samples_split=20
                )
                dt.fit(X_train, y_train)
                
                # Predict on validation set
                y_val_pred = dt.predict(X_val)
                r2 = r2_score(y_val, y_val_pred)
                
                dt_results[criterion] = {
                    'model': dt,
                    'r2_score': r2,
                    'predictions': y_val_pred
                }
                
                print(f"  {criterion:15s}: R^2 = {r2:.4f}")
                
                if r2 > best_score:
                    best_score = r2
                    best_criterion = criterion
                    
            except Exception as e:
                print(f"  {criterion:15s}: Error - {str(e)}")
        
        print(f"\nBest criterion: {best_criterion} (R^2 = {best_score:.4f})")
        
        if best_criterion is not None:
            self.models['decision_tree'] = dt_results[best_criterion]['model']
        self.results['decision_tree'] = dt_results
        
        return self.models.get('decision_tree'), best_criterion
    
    def train_svr(self, X_train, X_val, y_train, y_val):
        """Train Support Vector Regression"""
        print("\n" + "=" * 60)
        print("STEP 7: SUPPORT VECTOR REGRESSION")
        print("=" * 60)
        
        # Try different kernels
        kernels = ['rbf', 'linear', 'poly']
        best_kernel = None
        best_score = -np.inf
        svr_results = {}
        
        print("Testing different kernels for SVR:")
        
        for kernel in kernels:
            try:
                # Train SVR
                svr = SVR(kernel=kernel, C=1.0, gamma='scale')
                svr.fit(X_train, y_train)
                
                # Predict on validation set
                y_val_pred = svr.predict(X_val)
                r2 = r2_score(y_val, y_val_pred)
                
                svr_results[kernel] = {
                    'model': svr,
                    'r2_score': r2,
                    'predictions': y_val_pred
                }
                
                print(f"  {kernel:10s}: R^2 = {r2:.4f}")
                
                if r2 > best_score:
                    best_score = r2
                    best_kernel = kernel
                    
            except Exception as e:
                print(f"  {kernel:10s}: Error - {str(e)}")
        
        print(f"\nBest kernel: {best_kernel} (R^2 = {best_score:.4f})")
        
        if best_kernel is not None:
            self.models['svr'] = svr_results[best_kernel]['model']
        self.results['svr'] = svr_results
        
        return self.models.get('svr'), best_kernel
    
    def train_additional_models(self, X_train, X_val, y_train, y_val):
        """Train additional regression models for comparison"""
        print("\n" + "=" * 60)
        print("STEP 8: ADDITIONAL MODELS (CHALLENGE)")
        print("=" * 60)
        
        additional_models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=RANDOM_STATE,
                max_depth=10
            ),
            'Linear Regression': LinearRegression()
        }
        
        for name, model in additional_models.items():
            try:
                model.fit(X_train, y_train)
                y_val_pred = model.predict(X_val)
                r2 = r2_score(y_val, y_val_pred)
                
                self.models[name.lower().replace(' ', '_')] = model
                self.results[name.lower().replace(' ', '_')] = {
                    'model': model,
                    'r2_score': r2,
                    'predictions': y_val_pred
                }
                
                print(f" {name:20s}: R^2 = {r2:.4f}")
                
            except Exception as e:
                print(f" Error - {name:20s}: {str(e)}")
    
    def evaluate_final_models(self, X_test, y_test):
        """Evaluate all models on test set"""
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 60)
        
        final_results = {}
        
        for model_name, model in self.models.items():
            try:
                y_test_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_test_pred)
                mse = mean_squared_error(y_test, y_test_pred)
                mae = mean_absolute_error(y_test, y_test_pred)
                
                final_results[model_name] = {
                    'R^2 Score': r2,
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': np.sqrt(mse)
                }
                
                print(f"\n{model_name.upper().replace('_', ' ')}:")
                print(f"  R^2 Score: {r2:.4f}")
                print(f"  RMSE:     {np.sqrt(mse):.4f}")
                print(f"  MAE:      {mae:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
        
        return final_results
    
    def create_visualizations(self, final_results):
        """Create visualizations of results"""
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        # Extract R^2 scores for plotting
        model_names = list(final_results.keys())
        r2_scores = [final_results[name]['R^2 Score'] for name in model_names]
        
        # Create bar plot of R^2 scores
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        bars = plt.bar(range(len(model_names)), r2_scores, color=colors[:len(model_names)])
        plt.xlabel('Models')
        plt.ylabel('R^2 Score')
        plt.title('Model Performance Comparison (R^2 Score)')
        plt.xticks(range(len(model_names)), [name.replace('_', '\n') for name in model_names], rotation=0)
        plt.ylim(0, max(r2_scores) * 1.1)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, r2_scores)):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{score:.3f}', ha='center', va='bottom')
        
        # Create RMSE comparison
        plt.subplot(1, 2, 2)
        rmse_scores = [final_results[name]['RMSE'] for name in model_names]
        bars2 = plt.bar(range(len(model_names)), rmse_scores, color=colors[:len(model_names)])
        plt.xlabel('Models')
        plt.ylabel('RMSE')
        plt.title('Model Performance Comparison (RMSE)')
        plt.xticks(range(len(model_names)), [name.replace('_', '\n') for name in model_names], rotation=0)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars2, rmse_scores)):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(rmse_scores)*0.01,
                     f'{score:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations created and saved as 'model_comparison.png'")

def main():
    """Main execution function"""
    print("PRODUCT DEMAND FORECASTING MINI PROJECT")
    print("=" * 60)
    
    # Initialize the forecasting pipeline
    forecaster = ProductDemandForecasting()
    
    # Step 1: Load Dataset
    # NOTE: You need to download the dataset from Kaggle first!
    dataset_path = "Historical Product Demand.csv"  # Adjust this path as needed
    
    if not forecaster.load_data(dataset_path):
        print("\nTO GET STARTED:")
        print("1. Download the dataset from: https://www.kaggle.com/felixzhao/productdemandforecasting")
        print("2. Place the CSV file in the same directory as this script")
        print("3. Update the 'dataset_path' variable if needed")
        print("4. Run this script again")
        return
    
    # Step 2: Explore and clean data
    forecaster.explore_data()
    
    # Step 3: Feature selection and preprocessing
    X, y = forecaster.feature_selection()
    
    # Step 4: Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = forecaster.split_data(X, y)
    
    # Step 5: Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = forecaster.preprocess_features(
        X_train, X_val, X_test
    )
    
    # Store processed data
    forecaster.X_train = X_train_scaled
    forecaster.X_val = X_val_scaled
    forecaster.X_test = X_test_scaled
    forecaster.y_train = y_train
    forecaster.y_val = y_val
    forecaster.y_test = y_test
    
    # Step 6: Train Decision Tree with different criteria
    dt_model, best_dt_criterion = forecaster.train_decision_tree(
        X_train_scaled, X_val_scaled, y_train, y_val
    )
    
    # Step 7: Train SVR
    svr_model, best_svr_kernel = forecaster.train_svr(
        X_train_scaled, X_val_scaled, y_train, y_val
    )
    
    # Step 8: Train additional models (Challenge)
    forecaster.train_additional_models(X_train_scaled, X_val_scaled, y_train, y_val)
    
    # Final evaluation on test set
    final_results = forecaster.evaluate_final_models(X_test_scaled, y_test)
    
    # Create visualizations
    forecaster.create_visualizations(final_results)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"Decision Tree (Best Criterion: {best_dt_criterion})")
    if 'decision_tree' in final_results:
        print(f"   R^2 Score: {final_results['decision_tree']['R^2 Score']:.4f}")
    
    print(f"SVR (Best Kernel: {best_svr_kernel})")
    if 'svr' in final_results:
        print(f"   R^2 Score: {final_results['svr']['R^2 Score']:.4f}")
    
    # Find best performing model
    if final_results:
        best_model = max(final_results.keys(), key=lambda x: final_results[x]['R^2 Score'])
        print(f"\nBest Performing Model: {best_model.upper().replace('_', ' ')}")
        print(f"   R^2 Score: {final_results[best_model]['R^2 Score']:.4f}")
    
    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nDELIVERABLES CREATED:")
    print("Multiple R^2-squared measures for different Decision Tree criteria")
    print("R^2-squared measure for SVR output")
    print("Additional regression models for comparison")
    print("Model performance visualizations")
    print("Complete analysis pipeline")

if __name__ == "__main__":
    main()