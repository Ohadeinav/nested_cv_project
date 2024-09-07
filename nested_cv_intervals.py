import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

class LinearRegressorWithNestedCV:
    def __init__(self, k_outer=5, k_inner=5, quantiles=[0.7, 0.8, 0.9, 0.95]):
        self._k_outer = k_outer  # Outer loop for evaluation
        self._k_inner = k_inner  # Inner loop for model selection
        self._model = LinearRegression()
        self._quantiles = quantiles
        self._all_errors = []
        self._all_intervals = {}
        self._miscoverage_rates = {}

    def run_on_data(self, X, y, n_repetitions=100):
        """
        Implements nested cross-validation following Algorithm 1.
        It estimates prediction error and MSE using the outer and inner cross-validation loops.
        """
        all_errors = []  # Collect errors across all repetitions
        a_list = []  # List to store a terms
        b_list = []  # List to store b terms

        # Repeat the nested cross-validation n_repetitions times
        for _ in tqdm(range(n_repetitions), desc="Nested CV repetitions"):
            outer_kf = KFold(n_splits=self._k_outer, shuffle=True, random_state=None)

            for train_index, test_index in outer_kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Inner cross-validation to estimate the in-sample error
                inner_errors = self.inner_crossval(X_train, y_train)

                # Train the model on the full training data (excluding test set)
                self._model.fit(X_train, y_train)

                # Evaluate the model on the outer test set
                y_test_pred = self._model.predict(X_test)
                outer_error = mean_squared_error(y_test, y_test_pred)

                # Store the outer test error
                self._all_errors.append(outer_error)

                # Compute (a) and (b) terms
                a_term = (np.mean(inner_errors) - outer_error) ** 2
                b_term = np.var(outer_error) / len(test_index)

                a_list.append(a_term)
                b_list.append(b_term)

        # Final MSE and error estimates
        MSE_estimate = np.mean(a_list) - np.mean(b_list)  # Plug-in estimator
        mean_error = np.mean(self._all_errors)

        # Compute confidence intervals based on quantiles
        self._all_intervals = self.compute_confidence_intervals(self._all_errors)

        return mean_error, MSE_estimate

    def inner_crossval(self, X, y):
        """
        Inner cross-validation loop to estimate the in-sample error (e_in).
        """
        inner_kf = KFold(n_splits=self._k_inner, shuffle=True, random_state=None)
        inner_errors = []

        for inner_train_index, inner_val_index in inner_kf.split(X):
            X_inner_train, X_inner_val = X[inner_train_index], X[inner_val_index]
            y_inner_train, y_inner_val = y[inner_train_index], y[inner_val_index]

            # Train on inner training set
            self._model.fit(X_inner_train, y_inner_train)

            # Validate on the inner validation set
            y_inner_val_pred = self._model.predict(X_inner_val)
            inner_error = mean_squared_error(y_inner_val, y_inner_val_pred)

            inner_errors.append(inner_error)

        return inner_errors

    def compute_confidence_intervals(self, errors):
        """
        Compute confidence intervals for the outer-loop errors based on quantiles.
        """
        n = len(errors)
        errors = np.array(errors)

        # Calculate mean and standard error
        mean_error = np.mean(errors)
        std_error = np.std(errors, ddof=1) / np.sqrt(n)

        confidence_intervals = {}

        for level in self._quantiles:
            z_score = stats.norm.ppf((1 + level) / 2)  # Calculate Z-score

            # Calculate the margin of error
            margin_of_error = z_score * std_error

            # Calculate the confidence interval
            ci_lower = mean_error - margin_of_error
            ci_upper = mean_error + margin_of_error

            confidence_intervals[level] = (ci_lower, ci_upper)

        return confidence_intervals

    def plot_graph(self):
        # Extract the quantiles and intervals
        quantiles = list(self._all_intervals.keys())
        lower_bounds = [self._all_intervals[q][0] for q in quantiles]
        upper_bounds = [self._all_intervals[q][1] for q in quantiles]

        plt.figure(figsize=(10, 6))

        # Plot confidence intervals as horizontal lines for each quantile
        for i, q in enumerate(quantiles):
            plt.hlines(lower_bounds[i], q - 0.02, q + 0.02, color='blue', linestyles='solid', label='Lower Bound' if i == 0 else "")
            plt.hlines(upper_bounds[i], q - 0.02, q + 0.02, color='red', linestyles='solid', label='Upper Bound' if i == 0 else "")

        # Plot all error points per quantile, using transparency (alpha) to show density
        for i, q in enumerate(quantiles):
            jittered_q = q + np.random.uniform(-0.01, 0.01, size=len(self._all_errors))  # Small jitter to avoid exact overlap
            plt.scatter(jittered_q, self._all_errors, color='green', s=5, alpha=0.4, label='Test Errors' if i == 0 else "")

        # Annotate miscoverage rates below each quantile
        for i, q in enumerate(quantiles):
            miscoverage_rate = self._miscoverage_rates.get(q, 0)  # Replace with actual miscoverage calculation
            plt.text(q, lower_bounds[i] - 0.05, f'Miscoverage: {miscoverage_rate:.2f}', ha='center', va='top', fontsize=9, color='black')

        # Formatting the plot
        plt.xlabel('Quantiles')
        plt.ylabel('Error / Interval Bounds')
        plt.title('Test Errors and Confidence Intervals across Quantiles')

        # Move the legend to the upper right outside the plot
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        plt.grid(True)
        plt.tight_layout()  # Adjust layout so the legend doesn't overlap

        plt.show()


class CvIntervalsTest:
    def __init__(self, n_repetitions=100, k_outer=5, k_inner=5):
        self._n_repetitions = n_repetitions
        self._k_outer = k_outer
        self._k_inner = k_inner

    def run(self):
        # Generate data
        X, y, _ = generate_linear_data(n_samples=1000, n_features=5, noise=0.34)

        # Run regressor with nested cross-validation
        regressor = LinearRegressorWithNestedCV(k_outer=self._k_outer, k_inner=self._k_inner)
        mean_error, MSE_estimate = regressor.run_on_data(X, y, n_repetitions=self._n_repetitions)

        print(f"Estimated Prediction Error: {mean_error}")
        print(f"Estimated MSE: {MSE_estimate}")

        regressor.plot_graph()


def generate_linear_data(n_samples=1000, n_features=5, noise=0.2):
    """
    Generates linear data with Gaussian noise for the regression problem.
    """
    X = np.random.randn(n_samples, n_features)
    true_coefficients = np.random.randn(n_features)
    y = X.dot(true_coefficients) + np.random.normal(loc=0, scale=1, size=n_samples) * noise
    return X, y, true_coefficients


def main():
    test = CvIntervalsTest(n_repetitions=1000, k_outer=5, k_inner=5)
    test.run()


if __name__ == "__main__":
    main()
