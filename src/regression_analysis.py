import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'real_estate_prices.csv')

def main():
    print("="*60)
    print("LINEAR REGRESSION ANALYSIS - REAL ESTATE PRICE PREDICTION")
    print("="*60)
    # Load the dataset
    data = pd.read_csv(DATA_PATH)
    print(f"Dataset Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")

    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)

    # Convert price to lakhs for regression
    data['Price_Lakhs'] = data['Price (in lakhs)'] / 100000

    # Check for missing values
    print("Missing values per column:")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found")

    # Encode categorical columns
    label_encoders = {}
    categorical_columns = [
        'Type',
        'Home Loan Available',
        'Parking Area',
        'Appliances Included',
        'Direction',
        'Location'
    ]
    for col in categorical_columns:
        le = LabelEncoder()
        data[f'{col}_Encoded'] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Feature set
    features = [
        'Type_Encoded', 'Floors', 'Home Loan Available_Encoded',
        'Number of Kitchens', 'Parking Area_Encoded', 'Number of Balconies',
        'Appliances Included_Encoded', 'Area (sq ft)', 'Length (meters)',
        'Breadth (meters)', 'Direction_Encoded', 'Bedrooms',
        'Age of House', 'Location_Encoded'
    ]
    X = data[features]
    y = data['Price_Lakhs']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    print("\n" + "="*50)
    print("LINEAR REGRESSION MODEL")
    print("="*50)

    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_train_pred = lr_model.predict(X_train_scaled)
    y_test_pred = lr_model.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("Model Performance Metrics:")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Training R² Score: {train_r2:.3f}")
    print(f"Test R² Score: {test_r2:.3f}")

    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': lr_model.coef_,
        'Abs_Coefficient': np.abs(lr_model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    print(f"\nIntercept: {lr_model.intercept_:.2f}")
    print("\nTop 10 Most Important Features (by coefficient magnitude):")
    print(feature_importance.head(10))

    # VISUALIZATIONS
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    fig = plt.figure(figsize=(20, 24))

    # 1. Actual vs Predicted Prices (Training)
    plt.subplot(4, 3, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.6, color='blue', label='Training Data')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted Prices (Training Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Price (Lakhs)')
    plt.ylabel('Predicted Price (Lakhs)')
    plt.legend()
    plt.text(0.05, 0.95, f'R² = {train_r2:.3f}',
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Actual vs Predicted Prices (Test)
    plt.subplot(4, 3, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.6, color='green', label='Test Data')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted Prices (Test Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Price (Lakhs)')
    plt.ylabel('Predicted Price (Lakhs)')
    plt.legend()
    plt.text(0.05, 0.95, f'R² = {test_r2:.3f}',
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 3. Residuals Plot (Training)
    plt.subplot(4, 3, 3)
    residuals_train = y_train - y_train_pred
    plt.scatter(y_train_pred, residuals_train, alpha=0.6, color='blue')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals Plot (Training Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Price (Lakhs)')
    plt.ylabel('Residuals')

    # 4. Residuals Plot (Test)
    plt.subplot(4, 3, 4)
    residuals_test = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals_test, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals Plot (Test Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Price (Lakhs)')
    plt.ylabel('Residuals')

    # 5. Feature Coefficients
    plt.subplot(4, 3, 5)
    top_features = feature_importance.head(10)
    colors = ['red' if x < 0 else 'blue' for x in top_features['Coefficient']]
    plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors)
    plt.title('Top 10 Feature Coefficients', fontsize=14, fontweight='bold')
    plt.ylabel('Features')
    plt.xlabel('Coefficient Value')
    plt.yticks(range(len(top_features)), top_features['Feature'], rotation=0)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # 6. Model Performance Comparison
    plt.subplot(4, 3, 6)
    metrics = ['RMSE', 'MAE', 'R²']
    train_metrics = [train_rmse, train_mae, train_r2]
    test_metrics = [test_rmse, test_mae, test_r2]
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, train_metrics, width, label='Training', alpha=0.8)
    plt.bar(x + width/2, test_metrics, width, label='Test', alpha=0.8)
    plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.xticks(x, metrics)
    plt.legend()
    for i, (train_val, test_val) in enumerate(zip(train_metrics, test_metrics)):
        plt.text(i - width/2, train_val + 0.01, f'{train_val:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, test_val + 0.01, f'{test_val:.3f}', ha='center', va='bottom')

    # 7. Distribution of Residuals (Training)
    plt.subplot(4, 3, 7)
    plt.hist(residuals_train, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution of Residuals (Training)', fontsize=14, fontweight='bold')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')

    # 8. Q-Q Plot for Residuals
    plt.subplot(4, 3, 8)
    stats.probplot(residuals_test, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals (Test Set)', fontsize=14, fontweight='bold')

    # 9. Feature Correlation with Target
    plt.subplot(4, 3, 9)
    cols_to_corr = [
        'Area (sq ft)', 'Bedrooms', 'Age of House', 'Floors',
        'Number of Kitchens', 'Number of Balconies', 'Price_Lakhs'
    ]
    correlation_data = data[cols_to_corr]
    corr_with_price = correlation_data.corr()['Price_Lakhs'].drop('Price_Lakhs').sort_values(key=abs, ascending=False)
    plt.barh(range(len(corr_with_price)), corr_with_price.values)
    plt.title('Feature Correlation with Price', fontsize=14, fontweight='bold')
    plt.xlabel('Correlation Coefficient')
    plt.yticks(range(len(corr_with_price)), corr_with_price.index)

    # 10. Prediction Error Distribution
    plt.subplot(4, 3, 10)
    prediction_errors = np.abs(y_test - y_test_pred)
    plt.hist(prediction_errors, bins=15, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    plt.xlabel('Absolute Error (Lakhs)')
    plt.ylabel('Frequency')
    mean_error = np.mean(prediction_errors)
    plt.axvline(mean_error, color='red', linestyle='--', label=f'Mean Error: {mean_error:.2f}')
    plt.legend()

    # 11. Actual vs Predicted (Combined)
    plt.subplot(4, 3, 11)
    plt.scatter(y_train, y_train_pred, alpha=0.6, color='blue', label='Training', s=30)
    plt.scatter(y_test, y_test_pred, alpha=0.6, color='green', label='Test', s=30)
    min_y = min(y.min(), y_train_pred.min(), y_test_pred.min())
    max_y = max(y.max(), y_train_pred.max(), y_test_pred.max())
    plt.plot([min_y, max_y], [min_y, max_y], 'r--', lw=2)
    plt.title('Actual vs Predicted (Combined)', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Price (Lakhs)')
    plt.ylabel('Predicted Price (Lakhs)')
    plt.legend()

    # 12. Learning Curve Simulation
    plt.subplot(4, 3, 12)
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    val_scores = []
    for size in train_sizes:
        size_idx = int(size * len(X_train_scaled))
        X_temp = X_train_scaled[:size_idx]
        y_temp = y_train.iloc[:size_idx]
        temp_model = LinearRegression()
        temp_model.fit(X_temp, y_temp)
        train_pred = temp_model.predict(X_temp)
        val_pred = temp_model.predict(X_test_scaled)
        train_scores.append(r2_score(y_temp, train_pred))
        val_scores.append(r2_score(y_test, val_pred))
    plt.plot(train_sizes * len(X_train), train_scores, 'o-', color='blue', label='Training Score')
    plt.plot(train_sizes * len(X_train), val_scores, 'o-', color='green', label='Validation Score')
    plt.title('Learning Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Training Set Size')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # DETAILED ANALYSIS AND INSIGHTS
    print("\n" + "="*50)
    print("DETAILED ANALYSIS AND INSIGHTS")
    print("="*50)
    n_samples, n_features = X_train_scaled.shape
    mse_residual = test_mse
    residual_std_error = np.sqrt(mse_residual)
    print(f"Residual Standard Error: {residual_std_error:.3f}")
    print(f"Degrees of Freedom: {n_samples - n_features - 1}")

    # Feature impact analysis
    feature_analysis = pd.DataFrame({
        'Feature': features,
        'Coefficient': lr_model.coef_,
        'Abs_Coefficient': np.abs(lr_model.coef_)
    })
    print("\nFeature Impact Analysis:")
    for idx, row in feature_analysis.sort_values('Abs_Coefficient', ascending=False).head(5).iterrows():
        impact = "increases" if row['Coefficient'] > 0 else "decreases"
        print(f"• {row['Feature']}: {impact} price by {abs(row['Coefficient']):.3f} lakhs per unit")

    # Prediction accuracy analysis
    print(f"\nPrediction Accuracy Analysis:")
    print(f"Mean Absolute Error: ₹{test_mae * 100000:.0f}")
    print(f"Root Mean Square Error: ₹{test_rmse * 100000:.0f}")
    print(f"Model explains {test_r2*100:.1f}% of price variance")

    errors = np.abs(y_test - y_test_pred)
    print(f"\nPrediction Errors:")
    print(f"• Mean Error: ₹{np.mean(errors) * 100000:.0f}")
    print(f"• Median Error: ₹{np.median(errors) * 100000:.0f}")
    print(f"• 90th Percentile Error: ₹{np.percentile(errors, 90) * 100000:.0f}")

    print(f"\nSample Predictions (First 5 Test Cases):")
    sample_comparison = pd.DataFrame({
        'Actual_Price': y_test.iloc[:5].values,
        'Predicted_Price': y_test_pred[:5],
        'Error': np.abs(y_test.iloc[:5].values - y_test_pred[:5]),
        'Error_Percentage': (np.abs(y_test.iloc[:5].values - y_test_pred[:5]) / y_test.iloc[:5].values * 100)
    })
    for idx, row in sample_comparison.iterrows():
        print(f"Property {idx+1}:")
        print(f" Actual: ₹{row['Actual_Price']*100000:.0f}, Predicted: ₹{row['Predicted_Price']*100000:.0f}")
        print(f" Error: ₹{row['Error']*100000:.0f} ({row['Error_Percentage']:.1f}%)")

    print("\n" + "="*60)
    print("LINEAR REGRESSION ANALYSIS COMPLETE")
    print("="*60)
    print("Key Findings:")
    print(f"• Model R² Score: {test_r2:.3f} (explains {test_r2*100:.1f}% of variance)")
    print(f"• Average prediction error: ₹{test_mae*100000:.0f}")
    print(f"• Most important feature: {feature_importance.iloc[0]['Feature']}")
    print("• All visualizations generated successfully!")

if __name__ == '__main__':
    main()
