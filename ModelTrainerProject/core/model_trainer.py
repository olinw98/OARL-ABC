import numpy as np
from scipy.stats import beta, skew, kurtosis
import matplotlib.pyplot as plt
import logging
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model:
    def __init__(self, name, fxn=None, **kwargs):
        self.name = name
        self.alpha = kwargs.get('alpha', 2)
        self.beta = kwargs.get('beta', 2)
        self.betadistr = [self.alpha, self.beta]
        self.parameters = []

        # Determine if a custom function is provided
        self.is_custom_function = fxn is not None

        # If no custom function is provided, use the default function
        self.fxn = fxn or (lambda params, inputs: sum(params) + sum(inputs))

        # If function_str is not provided, initialize it for dynamic generation
        self.function_str = kwargs.get('function_str', None)
        if self.function_str is None and not self.is_custom_function:
            self.function_str = ""

        self.fxn_value = None

    def add_parameter(self, parameter):
        # Ensure the parameter is valid before adding
        if parameter is None or not isinstance(parameter, Parameter):
            logger.error("invalid parameter. Must be a Parameter object.")
            raise ValueError("Invalid parameter")
        
        self.parameters.append(parameter)
        logger.info(f"Parameter '{parameter.name}' added to the model.")

        # Update function_str dynamically if using the default function and no custom function was provided
        if not self.is_custom_function:
            self.function_str = " + ".join(f"{param.name}" for param in self.parameters)

    def finalize_function_str(self):
        # Finalize function_str in case no parameters were added in the constructor
        if not self.is_custom_function and not self.function_str:
            self.function_str = " + ".join(f"{param.name}" for param in self.parameters)

    def update_fxn(self):
        if self.parameters:
            self.fxn_value = self.fxn(self.parameters)

    def sample_reward(self):
        reward = beta.rvs(self.alpha, self.beta)
        logger.debug(f"Sampled reward for {self.name}: {reward}")
        return reward

    def sample_fxn(self, inputs):
        try:
            if self.parameters:
                sampled_values = [p.sample() for p in self.parameters]
                result = self.fxn(*inputs, *sampled_values)
                logger.info(f"Sampled function result: {result}")
                return result
        except Exception as e:
            logger.error(f"Error during function sampling: {e}")
            raise RuntimeError(f"Function sampling failed: {e}")

    def calculate_error(self, prediction, actual_data):
        if prediction is None or np.any(np.isnan(prediction)):
            logger.error(f"Prediction is NaN or invalid for model '{self.name}'. Returning high error.")
            return np.inf  # Return a high error if the prediction is invalid
        
        error = np.abs((prediction - actual_data) / (actual_data + 1e-8)).mean()
        
        if np.isnan(error):
            logger.error(f"Calculated error is NaN for model '{self.name}'. Returning high error.")
            return np.inf  # Return a high error if the error calculation results in NaN
        
        logger.debug(f"Calculated error for {self.name}: {error}")
        return error

    def update_beta_distr(self, error, epsilon, min_alpha=0.1, min_beta=0.1, max_shift=5):
        if np.isnan(error) or np.isinf(error):
            logger.error(f"Invalid error value encountered for {self.name}: {error}. Skipping update.")
            return

        if error < epsilon:
            alpha_adjustment = 1 + 2 * (epsilon - error) / epsilon
            self.alpha = max(self.alpha + min(alpha_adjustment, max_shift), min_alpha)
        else:
            beta_adjustment = 1 + 2 * (error - epsilon) / epsilon
            self.beta = max(self.beta + min(beta_adjustment, max_shift), min_beta)
        
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            logger.warning(f"Invalid alpha encountered for {self.name}: {self.alpha}. Resetting to default.")
            self.alpha = 2.0

        if not np.isfinite(self.beta) or self.beta <= 0:
            logger.warning(f"Invalid beta encountered for {self.name}: {self.beta}. Resetting to default.")
            self.beta = 2.0
        
        self.betadistr = [self.alpha, self.beta]
        logger.info(f"Updated Beta distribution for {self.name}: alpha={self.alpha}, beta={self.beta}")

    def plot_beta_distribution(self):
        x = np.linspace(0, 1, 100)
        y = beta.pdf(x, self.alpha, self.beta)
        plt.plot(x, y, label=f'{self.name} (α={self.alpha:.2f}, β={self.beta:.2f})')

    def predict(self, inputs):
        """
        Predict output based on inputs and sampled parameters.
        """
        if not self.parameters:
            raise ValueError(f"Model '{self.name}' has no parameters defined.")
        
        # Sample parameters
        sampled_params = [param.sampleval for param in self.parameters]

        # Use the model's function to predict output
        return self.fxn(sampled_params, inputs)

    def __repr__(self):
        return f"Model(name={self.name}, alpha={self.alpha}, beta={self.beta}, beta distribution={self.betadistr}, fxn={self.fxn}, fxn_value={self.fxn_value})"

class Parameter:
    def __init__(self, name, model, shape, **kwargs):
        self.name = name
        self.model = model
        self.shape = shape
        self.value = None
        self.sampleval = None
        self.parameters = {}
        self._set_distribution_parameters(**kwargs)
        model.add_parameter(self)

    def _set_distribution_parameters(self, **kwargs):
        try:
            if self.shape == 'normal':
                self.mean = kwargs.get('mean', 0)
                self.stddev = kwargs.get('stddev', 1)
                if self.stddev <= 0:
                    raise ValueError(f"Standard deviation for parameter '{self.name}' must be positive.")
                self.value = self.mean
                self.sampleval = np.random.normal(self.mean, self.stddev)
                self.parameters = {'mean': self.mean, 'stddev': self.stddev}
            elif self.shape == 'uniform':
                self.low = kwargs.get('low', 0)
                self.high = kwargs.get('high', 1)
                if self.low >= self.high:
                    raise ValueError(f"'low' must be less than 'high' for uniform distribution in parameter '{self.name}'.")
                self.value = (self.low + self.high) / 2
                self.sampleval = np.random.uniform(self.low, self.high)
                self.parameters = {'low': self.low, 'high': self.high}
            elif self.shape == 'beta':
                self.alpha = kwargs.get('alpha', 2)
                self.beta = kwargs.get('beta', 2)
                if self.alpha <= 0 or self.beta <= 0:
                    raise ValueError(f"'alpha' and 'beta' must be positive for beta distribution in parameter '{self.name}'.")
                self.value = self.alpha / (self.alpha + self.beta)
                self.sampleval = np.random.beta(self.alpha, self.beta)
                self.parameters = {'alpha': self.alpha, 'beta': self.beta}
            elif self.shape == 'triangular':
                self.low = kwargs.get('low', 0)
                self.high = kwargs.get('high', 1)
                self.mode = kwargs.get('mode', 0.5)
                if not (self.low <= self.mode <= self.high):
                    raise ValueError(f"'mode' must be between 'low' and 'high' for triangular distribution in parameter '{self.name}'.")
                self.value = self.mode
                self.sampleval = np.random.triangular(self.low, self.mode, self.high)
                self.parameters = {'low': self.low, 'high': self.high, 'mode': self.mode}
            else:
                raise ValueError(f"Unsupported shape '{self.shape}' for parameter '{self.name}'.")
        
        except ValueError as e:
            logger.error(f"Error in defining parameter '{self.name}': {e}")
            raise

    def sample(self):
        try:
            if self.shape == 'normal':
                self.sampleval = np.random.normal(self.mean, self.stddev)
            elif self.shape == 'uniform':
                self.sampleval = np.random.uniform(self.low, self.high)
            elif self.shape == 'beta':
                self.sampleval = np.random.beta(self.alpha, self.beta)
            elif self.shape == 'triangular':
                self.sampleval = np.random.triangular(self.low, self.mode, self.high)
            
            logger.debug(f"Sampled value for {self.name}: {self.sampleval}")
            return self.sampleval
        except ValueError as e:
            logger.error(f"Error in sampling parameter '{self.name}': {e}")
            raise

    def plot_posterior_distribution(self, samples):
        # Calculate distribution shape parameters
        mean = np.mean(samples)
        stddev = np.std(samples)
        median = np.median(samples)
        skewness = skew(samples)
        kurtosis_value = kurtosis(samples)

        plt.figure(figsize=(10, 6))
        
        # Plot histogram of accepted samples
        plt.hist(samples, bins=20, density=True, alpha=0.5, color='g', label='Accepted Samples')

        # Overlay kernel density estimate (KDE) as the posterior distribution
        sns.kdeplot(samples, color='r', label='Posterior Distribution', linewidth=2)

        # Title with model and parameter name
        plt.title(f'{self.model.name}: {self.name} - Posterior Distribution\n'
                  f'Mean: {mean:.3f}, Stddev: {stddev:.3f}, Median: {median:.3f}')
        
        # Display statistics as text inside the plot
        stats_text = (f'Mean: {mean:.3f}\n'
                    f'Stddev: {stddev:.3f}\n'
                    f'Median: {median:.3f}\n'
                    f'Skewness: {skewness:.3f}\n'
                    f'Kurtosis: {kurtosis_value:.3f}')
        plt.text(0.98, 0.85, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.6))

        # Labels
        plt.xlabel('Parameter Value')
        plt.ylabel('Density')

        # Add legend
        plt.legend()

        # Show plot
        plt.show() # OR plt.show(block=False)

    def __repr__(self):
        return f"Parameter(name={self.name}, shape={self.shape}, value={self.value}, parameters={self.parameters})"

class BayesianModel(Model):
    def __init__(self, name, fxn=None, **kwargs):
        super().__init__(name, fxn, **kwargs)
        self.accepted_samples = {}  # Initialize as an empty dictionary

    def initialize_accepted_samples(self):
        """Initialize the accepted samples dictionary after parameters have been added."""
        self.accepted_samples = {param.name: [] for param in self.parameters}

    def sample_from_accepted(self, param_name, exploration_rate=0.2):
        """Sample from the accepted samples for the given parameter with occasional exploration."""
        if np.random.rand() < exploration_rate:
            logger.info(f"Exploration: Sampling {param_name} from the prior.")
            return self.sample_prior(param_name)
        elif self.accepted_samples[param_name]:
            return np.random.choice(self.accepted_samples[param_name])
        else:
            return self.sample_prior(param_name)
        
    def sample_prior(self, param_name):
        for param in self.parameters:
            if param.name == param_name:
                return param.sample()

    def calibrate_parameters_abc(self, inputs, actual_output, tolerance=0.1, update_threshold=100):
        tolerance = max(tolerance, 0.05)  # Prevent tolerance from dropping below 0.05
        sampled_params = []
        error = np.inf
        
        # Step 1: Sample from priors
        for param in self.parameters:
            sampled_value = param.sample()  # Sample a value from the prior distribution
            sampled_params.append(sampled_value)  # Store sampled parameter value
        
        # Step 2: Generate model prediction based on sampled parameters
        try:
            prediction = self.fxn(*inputs, *sampled_params)
        except Exception as e:
            logger.error(f"Error during function execution for model '{self.name}': {e}")
            return self.accepted_samples

        # Ensure valid prediction and error
        if np.any(np.isnan(prediction)):
            logger.error(f"Model '{self.name}' produced a NaN prediction; skipping this iteration.")
            return self.accepted_samples
        
        # Step 3: Calculate the error between prediction and actual data
        error = self.calculate_error(prediction, actual_output)
        if np.isnan(error):
            logger.error(f"Error calculation resulted in NaN for model '{self.name}'; skipping this iteration.")
            return self.accepted_samples
        
        logger.info(f"Calculated error for {self.name}: {error:.3f}")
        
        # Step 4: Accept or reject the sampled parameters
        if error < tolerance:
            logger.info(f"Accepted parameters for {self.name} with error {error:.3f}")
            for param in self.parameters:
                self.accepted_samples[param.name].append(param.sampleval)
            self.update_beta_distr(error, tolerance)
        else:
            logger.info(f"Rejected parameters for {self.name} with error {error:.3f}")
            self.update_beta_distr(error, tolerance)
        
        # Step 5: Update parameter distributions if enough samples have been accepted
        if all(len(samples) >= update_threshold for samples in self.accepted_samples.values()):
            logger.info(f"Updating parameter distributions for {self.name} based on accepted samples.")
            self.update_parameter_distributions()
            self.initialize_accepted_samples() # Reset the accepted samples after updating distributions

        return self.accepted_samples

    def update_parameter_distributions(self):
        """Update the parameter distributions based on accepted samples."""
        for param in self.parameters:
            accepted_values = self.accepted_samples[param.name]
            if param.shape == 'normal':
                # Update mean and stddev based on accepted samples
                param.mean = np.mean(accepted_values)
                param.stddev = max(np.std(accepted_values), 1e-3)  # Prevent stddev from being zero
                param.parameters['mean'] = param.mean
                param.parameters['stddev'] = param.stddev
                logger.info(f"Updated {param.name} to new normal distribution: mean={param.mean}, stddev={param.stddev}")
            elif param.shape == 'uniform':
                # Update low and high bounds based on accepted samples
                param.low = np.min(accepted_values)
                param.high = np.max(accepted_values)
                param.parameters['low'] = param.low
                param.parameters['high'] = param.high
                logger.info(f"Updated {param.name} to new uniform distribution: low={param.low}, high={param.high}")
            elif param.shape == 'beta':
                # Update alpha and beta based on accepted samples (method of moments)
                mean = np.mean(accepted_values)
                variance = np.var(accepted_values)
                if variance > 0 and 0 < mean < 1:
                    param.alpha = max(mean * ((mean * (1 - mean) / variance) - 1), 1e-3)
                    param.beta = max((1 - mean) * ((mean * (1 - mean) / variance) - 1), 1e-3)
                    param.parameters['alpha'] = param.alpha
                    param.parameters['beta'] = param.beta
                    logger.info(f"Updated {param.name} to new beta distribution: alpha={param.alpha}, beta={param.beta}")
                else:
                    logger.warning(f"Could not update beta distribution for {param.name} due to invalid mean or variance.")
            elif param.shape == 'triangular':
                # Update low, mode, and high based on accepted samples
                param.low = np.min(accepted_values)
                param.high = np.max(accepted_values)
                param.mode = np.median(accepted_values)
                param.parameters['low'] = param.low
                param.parameters['high'] = param.high
                param.parameters['mode'] = param.mode
                logger.info(f"Updated {param.name} to new triangular distribution: low={param.low}, mode={param.mode}, high={param.high}")
            else:
                logger.warning(f"Unsupported parameter shape '{param.shape}' for updating distribution.")
        
        logger.info(f"Parameter distributions updated for model '{self.name}'.")

class RLTrainer:
    def __init__(self, models, input_data, output_data, epsilon, num_episodes=500, reward_function=None):
        self.models = models
        self.input_data = input_data
        self.output_data = output_data
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.reward_function = reward_function or self.default_reward_function
        self.rewards_history = []

        # Initialize accepted samples for all models
        for model in self.models:
            model.initialize_accepted_samples()

    def default_reward_function(self, model):
        return model.sample_reward()

    def select_model(self, epsilon=0.15):
        if np.random.rand() < epsilon:
            selected_model = np.random.choice(self.models)  # Random exploration
            logger.info(f"Randomly selected model '{selected_model.name}' for exploration.")
        else:
            sampled_rewards = [self.reward_function(model) for model in self.models]
            selected_model = self.models[sampled_rewards.index(max(sampled_rewards))]
            logger.info(f"Selected model '{selected_model.name}' based on highest reward.")
        return selected_model

    def adaptive_tolerance(self, episode, initial_tolerance=0.5, decay_rate=0.005, warmup_episodes=20, max_tolerance=1.0):
        if episode < warmup_episodes:
            return initial_tolerance
        success_rate = max(sum([len(model.accepted_samples[param]) for model in self.models for param in model.accepted_samples]), 1) / (episode + 1)
        tolerance = initial_tolerance * np.exp(-decay_rate * episode * (1 - success_rate))
        tolerance = max(tolerance, 0.05)  # Minimum tolerance floor
        tolerance = min(tolerance, max_tolerance)  # Maximum tolerance ceiling
        return tolerance
    
    def train(self):
        for episode in range(self.num_episodes):
            # Randomly select a data point from the dataset
            data_index = np.random.randint(0, len(self.input_data))
            inputs = self.input_data[data_index]
            actual_output = self.output_data[data_index]

            model = self.select_model()  # Select a model for this episode
            
            # Step 1: Calibrate the selected model using ABC
            tolerance = self.adaptive_tolerance(episode, initial_tolerance=self.epsilon)
            posteriors = model.calibrate_parameters_abc(inputs, actual_output, tolerance=tolerance)

            # Step 2: Optionally re-sample parameter values for the next round
            for param in model.parameters:
                param.sampleval = model.sample_from_accepted(param.name)  # Resample from accepted posterior
            
            logger.info(f"Episode {episode + 1}/{self.num_episodes}: Model '{model.name}' calibrated.") # with tolerance {tolerance:.3f}. Accepted samples: {sum(len(s) for s in model.accepted_samples.values())}")

            # Optional: Early stopping if the model has converged (e.g., no significant changes in error)
            # Uncomment and adjust the criteria as needed
            # if self.has_converged(model):
            #     logger.info(f"Model '{model.name}' has converged. Stopping training early.")
            #     break

    def evaluate_best_model(self):
        model_errors = []
        model_rewards = []

        for model in self.models:
            avg_error = np.mean([abs(self.output_data[i] - model.sample_fxn(self.input_data[i])) for i in range(len(self.output_data))])
            model_errors.append(avg_error if np.isfinite(avg_error) else np.inf)

            expected_reward = model.alpha / (model.alpha + model.beta)
            model_rewards.append(expected_reward)

            logger.info(f"Model '{model.name}' - Avg Error: {avg_error:.3f}, Expected Reward: {expected_reward:.3f}")

        combined_scores = [
            0.5 * e + 0.5 * (1 - r) if np.isfinite(e) else np.inf for e, r in zip(model_errors, model_rewards)
        ]

        best_model_index = np.argmin(combined_scores)
        best_model = self.models[best_model_index]

        logger.info(f"Best Model: {best_model.name} with Combined Score: {combined_scores[best_model_index]:.3f}")
        return best_model

    def plot_beta_distributions(self):
        plt.figure(figsize=(15, 5))
        for i, model in enumerate(self.models):
            x = np.linspace(0, 1, 100)
            y = beta.pdf(x, model.alpha, model.beta)
            plt.plot(x, y, label=f'{model.name} (α={model.alpha:.2f}, β={model.beta:.2f})')
        plt.title('Beta Distributions of Models')
        plt.xlabel('Probability')
        plt.ylabel('Density')
        plt.legend()

    def plot_synthetic_vs_model_outputs(self, input_data, output_data):
        plt.figure(figsize=(10, 6))
        plt.plot(output_data, label='Actual Output Data', color='black', linewidth=2)

        for model in self.models:
            model_output = np.array([model.sample_fxn(input_data[i]) for i in range(len(input_data))])
            plt.plot(model_output, label=f'{model.name} Output', linestyle='--')

        plt.title('Comparison of Actual Output Data and Model Outputs')
        plt.xlabel('Sample Index')
        plt.ylabel('Output Value')
        plt.legend()

    def log_and_plot_posteriors(self):
        for model in self.models:
            logger.info(f"Posterior distributions for {model.name}:")
            for param in model.parameters:
                samples = model.accepted_samples.get(param.name, [])
                if samples:
                    logger.info(f"Plotting posterior distribution for {param.name} in model {model.name}.")
                    param.plot_posterior_distribution(samples)

    def plot_debugging_info(self):
        # Plot Beta Distributions
        self.plot_beta_distributions()

        # Plot Posterior Distributions
        self.log_and_plot_posteriors()

        # Plot Synthetic vs Model Outputs
        self.plot_synthetic_vs_model_outputs(self.input_data, self.output_data)

        # Display all plots at the end
        plt.show()

    def run(self):
        self.train()
        best_model = self.evaluate_best_model()
        self.plot_debugging_info()
        return best_model