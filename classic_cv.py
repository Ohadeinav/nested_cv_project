import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

class LinearRegressorWithClassicCV:
    def __init__(self, k, test_size=0.2, ci=[0.7, 0.8, 0.9, 0.95]):
        self._n_folds = k
        self._test_size = test_size
        self._model = LinearRegression()
        self._confidence_intervals = ci

    def run_on_data(self, X, y):
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self._test_size, random_state=42)

        # Cross-validation on the training set
        kf = KFold(n_splits=self._n_folds, shuffle=True, random_state=42)
        errors = []
        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            # Train the model on the training fold
            self._model.fit(X_train_fold, y_train_fold)

            # Validate the model on the validation fold
            y_pred = self._model.predict(X_val_fold)
            fold_error = mean_squared_error(y_val_fold, y_pred)
            errors.append(fold_error)

        # Calculate MSE on the test set (final evaluation)
        y_test_pred = self._model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)

        # Compute confidence intervals for the cross-validation errors
        confidence_intervals = self.compute_confidence_intervals(errors, self._confidence_intervals)

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
            z_score = stats.norm.ppf((1 + level) / 2) # TODO: fix this - Z-Score of 1-(alpha/2)

            # Calculate the margin of error
            margin_of_error = z_score * std_error

            # Calculate the confidence interval
            ci_lower = mean_error - margin_of_error
            ci_upper = mean_error + margin_of_error

            confidence_intervals[level] = (ci_lower, ci_upper)

        return confidence_intervals


class CvIntervalsTest:

    def __init__(self, n_simulations=1000, quantiles=[0.7, 0.8, 0.9, 0.95]):
        np.random.seed(42)  # For reproducibility
        self._n_simulations = n_simulations
        self._quantiles = quantiles
        # Arrays to store results
        self._all_errors = []
        self._all_intervals = []
        self._miscoverage_rates = {}

    def _compute_miscoverage_rates(self):
        miscoverage_rates = {}
        for q in self._quantiles:
            within_interval = sum(
                intervals[q][0] <= error <= intervals[q][1]  for error, intervals in zip(self._all_errors, self._all_intervals)
            )
            miscoverage_rates[q] = 1 - (within_interval / self._n_simulations)
        return miscoverage_rates

    def run(self):
        for _ in tqdm(range(self._n_simulations)):
            # Generate data
            X, y, _ = generate_linear_data(n_samples=100000, n_features=5, noise=0.1)

            # Run regressor
            regressor = LinearRegressorWithClassicCV(k=5, test_size=0.2)
            test_mse, confidence_intervals = regressor.run_on_data(X, y)

            self._all_errors.append(test_mse)
            self._all_intervals.append(confidence_intervals)

        # Calculate miscoverage rates
        self._miscoverage_rates = self._compute_miscoverage_rates()

    def plot_graph(self):

        plt.figure(figsize=(14, 8))

        # Plotting
        plt.figure(figsize=(14, 8))

        # Plot quantile bounds
        lower_scatter = None
        upper_scatter = None
        lower_line = None
        upper_line = None

        for i, q in enumerate(self._quantiles):
            lower_bounds = [intervals[q][0] for intervals in self._all_intervals]
            upper_bounds = [intervals[q][1] for intervals in self._all_intervals]

            # Calculate jittered x-positions
            x_jitter = np.random.normal(q, 0.002, len(lower_bounds))

            lower_scatter = plt.scatter(x_jitter, lower_bounds, color='blue', alpha=0.1, s=20)
            upper_scatter = plt.scatter(x_jitter, upper_bounds, color='red', alpha=0.1, s=20)

            # Plot mean bounds with wider lines
            mean_lower = np.mean(lower_bounds)
            mean_upper = np.mean(upper_bounds)
            lower_line, = plt.plot([q - 0.01, q + 0.01], [mean_lower, mean_lower], color='blue', linewidth=3)
            upper_line, = plt.plot([q - 0.01, q + 0.01], [mean_upper, mean_upper], color='red', linewidth=3)

            # Annotate with miscoverage rate
            plt.text(q, plt.gca().get_ylim()[1], f'{self._miscoverage_rates[q]:.3f}',
                     ha='center', va='bottom', rotation=45)

        plt.title('Quantile Bounds and Miscoverage Rates', fontsize=16)
        plt.xlabel('Quantiles', fontsize=14)
        plt.ylabel('Mean Squared Error', fontsize=14)
        plt.xticks(self._quantiles, [f'{q:.3f}' for q in self._quantiles], rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add corrected legend
        plt.legend([lower_scatter, upper_scatter, lower_line, upper_line],
                   ['Error point below the interval center', 'Error point above the interval center',
                    'Lower Bound', 'Upper Bound'],
                   loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

        plt.tight_layout()
        plt.show()

        # Print miscoverage rates
        print("\nMiscoverage Rates:")
        for q, rate in self._miscoverage_rates.items():
            print(f"{q * 100}% Quantile: {rate:.3f}")


def generate_linear_data(n_samples=100000, n_features=5, noise=0.1):
    X = np.random.randn(n_samples, n_features)
    true_coefficients = np.random.randn(n_features)
    y = X.dot(true_coefficients) + np.random.randn(n_samples) * noise
    return X, y, true_coefficients



def main():

    test = CvIntervalsTest(1, [0.7, 0.8, 0.9, 0.95])
    test.run()
    test.plot_graph()


if __name__ == "__main__":
    main()

