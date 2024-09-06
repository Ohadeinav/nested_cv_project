import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt

class LinearRegressorWithClassicCV:
    def __init__(self, k, test_size=0.2):
        self.n_folds = k
        self.test_size = test_size
        self.model = LinearRegression()

    def run_on_data(self, X, y):
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)

        # Cross-validation on the training set
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        errors = []
        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            # Train the model on the training fold
            self.model.fit(X_train_fold, y_train_fold)

            # Validate the model on the validation fold
            y_pred = self.model.predict(X_val_fold)
            fold_error = mean_squared_error(y_val_fold, y_pred)
            errors.append(fold_error)

        # Calculate MSE on the test set (final evaluation)
        y_test_pred = self.model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)

        # Compute confidence intervals for the cross-validation errors
        confidence_levels = [0.80, 0.85, 0.90, 0.95, 0.99, 0.999]  # 80% and higher
        confidence_intervals = self.compute_confidence_intervals(errors, confidence_levels)

        return test_mse, confidence_intervals

    def compute_confidence_intervals(self, errors, confidence_levels):
        n = len(errors)
        errors = np.array(errors)

        # Calculate mean and standard error
        mean_error = np.mean(errors)
        std_error = np.std(errors, ddof=1) / np.sqrt(n)

        confidence_intervals = {}

        for level in confidence_levels:
            # Calculate the Z-score for the given confidence level using SciPy
            z_score = stats.norm.ppf((1 + level) / 2)

            # Calculate the margin of error
            margin_of_error = z_score * std_error

            # Calculate the confidence interval
            ci_lower = mean_error - margin_of_error
            ci_upper = mean_error + margin_of_error

            confidence_intervals[level] = (ci_lower, ci_upper)

        return confidence_intervals


def generate_linear_data(n_samples=1000, n_features=5, noise=0.1):
    X = np.random.randn(n_samples, n_features)
    true_coefficients = np.random.randn(n_features)
    y = X.dot(true_coefficients) + np.random.randn(n_samples) * noise
    return X, y, true_coefficients


def run_test():
    # Generate synthetic data
    X, y, true_coefficients = generate_linear_data(n_samples=1000, n_features=5, noise=0.1)

    # Initialize and run the regressor
    regressor = LinearRegressorWithClassicCV(k=5, test_size=0.2)
    test_mse, confidence_intervals = regressor.run_on_data(X, y)

    print(f"Test MSE: {test_mse}")
    for level, interval in confidence_intervals.items():
        print(f"{level * 100}% Confidence Interval: {interval}")

    # Calculate true error distribution
    n_simulations = 1000
    errors = []
    for _ in range(n_simulations):
        X_sim, y_sim, _ = generate_linear_data(n_samples=1000, n_features=5, noise=0.1)
        X_train, X_test, y_train, y_test = train_test_split(X_sim, y_sim, test_size=0.2)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        errors.append(mean_squared_error(y_test, y_pred))

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot error distribution
    plt.hist(errors, bins=50, density=True, alpha=0.7, color='skyblue')
    plt.axvline(test_mse, color='red', linestyle='dashed', linewidth=2, label='Observed Test MSE')

    # Plot confidence intervals
    y_height = plt.gca().get_ylim()[1] * 0.9
    for level, (ci_lower, ci_upper) in confidence_intervals.items():
        plt.plot([ci_lower, ci_upper], [y_height, y_height], 'g-', linewidth=2,
                 label=f'{level * 100}% CI' if level == 0.95 else '')
        plt.text(ci_upper, y_height, f'{level * 100}%', verticalalignment='bottom')

    plt.title('Error Distribution with Confidence Intervals')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_test()