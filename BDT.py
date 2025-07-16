import numpy as np
import pandas as pd
import scipy.stats as stats
import random
import logging
from collections import defaultdict
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import json
import matplotlib.pyplot as plt
import seaborn as sns
import ross as rs
from collections import deque
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend, coherence, correlate, find_peaks, medfilt
import psutil
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            elif self.shape == 'constant':
                self.value = kwargs.get('value', 0)
                self.sampleval = self.value
                self.parameters = {'value': self.value}
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
            elif self.shape == 'constant':
                self.sampleval = self.value
            logger.debug(f"Sampled value for {self.name}: {self.sampleval}")
            return self.sampleval
        except ValueError as e:
            logger.error(f"Error in sampling parameter '{self.name}': {e}")
            raise

    def sample_final(self):
        try:
            if self.shape == 'normal':
                self.sampleval = self.mean
            elif self.shape == 'constant':
                self.sampleval = self.value
            logger.debug(f"Final sampled value for {self.name}: {self.sampleval}")
            return self.sampleval
        except ValueError as e:
            logger.error(f"Error in sampling parameter '{self.name}': {e}")
            raise

    def update_parameter_distribution(self, new_mean, new_stddev):
        if self.shape == 'normal':
            self._set_distribution_parameters(mean=new_mean, stddev=new_stddev)

    def __repr__(self):
        return f"Parameter(name={self.name}, shape={self.shape}, value={self.value}, parameters={self.parameters})"

class RotorModel:
    def __init__(self, name, shaft_lengths, disk_locations, inputs, **kwargs):
        self.name = name
        self.shaft_lengths = shaft_lengths
        self.disk_locations = disk_locations
        self.inputs = inputs
        self.alpha = kwargs.get('alpha', 2)
        self.beta = kwargs.get('beta', 2)
        self.betadistr = [self.alpha, self.beta]
        self.parameters = {
            'kxx_left': None, 'kxy_left': None, 'kyx_left': None, 'kyy_left': None,
            'cxx_left': None, 'cxy_left': None, 'cyx_left': None, 'cyy_left': None,
            'kxx_right': None, 'kxy_right': None, 'kyx_right': None, 'kyy_right': None,
            'cxx_right': None, 'cxy_right': None, 'cyx_right': None, 'cyy_right': None,
            'left_rotor_imb_mag': None, 'right_rotor_imb_mag': None, 'left_rotor_imb_phase': None, 'right_rotor_imb_phase': None,
            'rad_stiff': None, 'bend_stiff': None, 'x_misalign': None, 'y_misalign': None, 'misalign_angle': None,
            'speed': None
        }

        # Accepted posterior samples for each parameter
        self.accepted_samples = {param: [] for param in self.parameters.keys()}

        # Rotor materials
        self.steel = rs.Material(name="Steel", rho=7800, E=2.1e11, Poisson=0.29)
        self.aluminum = rs.Material(name="Aluminum", rho=2700, E=6.895e10, Poisson=0.33)

        self.rotor = None  # Placeholder for rotor model
        self.frequency_response = None

    def add_parameter(self, parameter):
        """
        Adds a parameter to the rotor model after ensuring its validity.
        """
        if parameter.name not in self.parameters:
            raise ValueError(f"Parameter name '{parameter.name}' is not valid for RotorModel.")

        if self.parameters[parameter.name] is not None:
            logging.warning(f"Parameter '{parameter.name}' is already initialized for RotorModel '{self.name}'. Overwriting.")

        self.parameters[parameter.name] = parameter
        logging.info(f"Parameter '{parameter.name}' added to RotorModel '{self.name}'.")

    def sample_parameters(self):
        """
        Samples values for all initialized parameters and returns a dictionary of sampled values.
        """
        sampled_params = {}
        for param_name, param in self.parameters.items():
            if param is not None:
                sampled_value = param.sample()
                sampled_params[param_name] = sampled_value
                logging.info(f"Sampled {param_name}: {sampled_value}")
            else:
                raise ValueError(f"Parameter '{param_name}' has not been initialized in RotorModel '{self.name}'.")

        return sampled_params

    def sample_parameters_final(self):
        """
        Samples values for all initialized parameters and returns a dictionary of sampled values.
        """
        sampled_params = {}
        for param_name, param in self.parameters.items():
            if param is not None:
                sampled_value = param.sample_final()
                sampled_params[param_name] = sampled_value
                logging.info(f"Sampled {param_name}: {sampled_value}")
            else:
                raise ValueError(f"Parameter '{param_name}' has not been initialized in RotorModel '{self.name}'.")

        return sampled_params

    def check_parameters_initialized(self):
        """
        Check if all required parameters have been initialized.
        Raises an error if any parameters are None.
        """
        for param_name, param in self.parameters.items():
            if param is None:
                raise ValueError(f"Parameter '{param_name}' has not been initialized in RotorModel '{self.name}'.")
        logging.info("All parameters have been successfully initialized.")

    def build_rotor(self, sampled_params):
        """
        Assemble the rotor using sampled bearing parameters.
        """
        # Ensure all parameters have been initialized before building the rotor
        self.check_parameters_initialized()

        # Create shaft elements
        id_shaft = 0            # meters
        od_shaft = 0.01905      # meters

        shaft_elems = []
        node_locs = [0.0]
        len_track = 0.0
        for length in self.shaft_lengths:
            shaft_elems.append(
                rs.ShaftElement6DoF(
                     L = length,
                     idl = id_shaft,
                     odl = od_shaft,
                     material = self.steel,
                     shear_effects = True,
                     rotary_inertia = True,
                     gyroscopic = True,
                )
            )
            node_locs.append(len_track + length)
            len_track += length

        # for length in self.shaft_lengths:
        #     shaft_elems.append(
        #         rs.ShaftElement(
        #              L = length,
        #              idl = id_shaft,
        #              odl = od_shaft,
        #              material = self.steel,
        #              shear_effects = True,
        #              rotary_inertia = True,
        #              gyroscopic = True,
        #         )
        #     )
        #     node_locs.append(len_track + length)
        #     len_track += length

        # define disk nodes
        disk_nodes = [min(range(len(node_locs)), key=lambda i: abs(node_locs[i] - loc)) for loc in self.disk_locations]

        id_disk = od_shaft
        od_disk = 0.1524        # meters
        width = 0.0157         # meters

        disks = [
            rs.DiskElement6DoF(
                n = disk_nodes[i],
                m = 1.679,  # kg
                Id = 0.00625,
                Ip = 0.1205
            )
            for i in range(len(disk_nodes))
        ]

        # disks = [
        #     rs.DiskElement.from_geometry(
        #         n = disk_nodes[i],
        #         material = self.aluminum,
        #         width = width,
        #         i_d = id_disk,
        #         o_d = od_disk,
        #     )
        #     for i in range(len(disk_nodes))
        # ]

        # specify supports
        self.n_b0 = 0

        # Create the bearing elements with the full set of coefficients
        bearing_left = rs.BearingElement6DoF(n=self.n_b0, kxx=sampled_params['kxx_left'], kxy=sampled_params['kxy_left'],
                                         kyx=sampled_params['kyx_left'], kyy=sampled_params['kyy_left'],
                                         cxx=sampled_params['cxx_left'], cxy=sampled_params['cxy_left'],
                                         cyx=sampled_params['cyx_left'], cyy=sampled_params['cyy_left'])

        bearing_right = rs.BearingElement6DoF(n=len(self.shaft_lengths),
                                          kxx=sampled_params['kxx_right'], kxy=sampled_params['kxy_right'],
                                          kyx=sampled_params['kyx_right'], kyy=sampled_params['kyy_right'],
                                          cxx=sampled_params['cxx_right'], cxy=sampled_params['cxy_right'],
                                          cyx=sampled_params['cyx_right'], cyy=sampled_params['cyy_right'])

        # Assemble the rotor with shaft, disk, and bearing elements
        self.rotor = rs.Rotor(shaft_elems, disks, [bearing_left, bearing_right])
        logging.info(f"Rotor model '{self.name}' built successfully.")

    def calculate_acc(self, disp_arr, time_arr):
        time_deltas = np.diff(time_arr)
        vel_arr = np.diff(disp_arr) / time_deltas
        acc_arr = np.diff(vel_arr) / time_deltas[:-1]
        return acc_arr / 9.81 # Converted to g's

    def compute_fft_with_window(self, vib_arr, samp_int, detrend_data = True, window_data = True):
        if detrend_data:
            vib_arr = detrend(vib_arr)

        # Remove DC offset
        vib_arr = vib_arr - np.mean(vib_arr)

        # Apply the Hann window
        if window_data:
            window = np.hanning(len(vib_arr))
            vib_arr = vib_arr * window

        # Perform FFT on the windowed signal
        fft = rfft(vib_arr)

        # Normalize using RMS of the window
        rms_window = np.sqrt(np.mean(window**2))
        amplitudes = (2 * np.abs(fft) / len(vib_arr)) / rms_window

        frequencies = rfftfreq(len(vib_arr), samp_int)

        # Filter out frequencies below 5 Hz and above 100 Hz (effects of minor physical imperfections)
        mask = (frequencies >= 5) & (frequencies <= 100)
        frequencies = frequencies[mask]
        amplitudes = amplitudes[mask]
        phases = np.angle(fft)[mask]

        return {'freq': frequencies, 'amp': amplitudes, 'phase': phases}

    def plot_sim_responses(self, time_results, frequency_results, sampling_rate = 20000):
        # Extract Time-Domain and Frequency-Domain Responses for Both Bearings
        left_time_acc = time_results['probe_x_0']['acc']  # Left bearing acceleration (x-axis)
        left_time_acc = left_time_acc - np.mean(left_time_acc)
        right_time_acc = time_results['probe_x_1']['acc']  # Right bearing acceleration (x-axis)
        right_time_acc = right_time_acc - np.mean(right_time_acc)

        left_freqs = frequency_results['probe_x_0']['freq']  # Left bearing frequencies
        left_amps = frequency_results['probe_x_0']['amp']  # Left bearing amplitudes

        right_freqs = frequency_results['probe_x_1']['freq']  # Right bearing frequencies
        right_amps = frequency_results['probe_x_1']['amp']  # Right bearing amplitudes

        # Generate 4 Subplots: Time-Domain and Frequency-Domain for Left and Right Bearings
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Predicted Time and Frequency Domain Analysis', fontsize=20)

        # Plot Time-Domain for Left Bearing
        axs[0, 0].plot(np.arange(len(left_time_acc)) / sampling_rate, left_time_acc)
        axs[0, 0].set_title('Left Bearing - Time Domain')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Acceleration (g)')
        axs[0, 0].grid(True)

        # Plot Time-Domain for Right Bearing
        axs[0, 1].plot(np.arange(len(right_time_acc)) / sampling_rate, right_time_acc)
        axs[0, 1].set_title('Right Bearing - Time Domain')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Acceleration (g)')
        axs[0, 1].grid(True)

        # Plot Frequency-Domain for Left Bearing
        axs[1, 0].plot(left_freqs, left_amps)
        axs[1, 0].set_xlim(1, max(left_freqs))
        axs[1, 0].set_title('Left Bearing - Frequency Domain')
        axs[1, 0].set_xlabel('Frequency (Hz)')
        axs[1, 0].set_ylabel('Amplitude (g)')
        axs[1, 0].grid(True)

        # Plot Frequency-Domain for Right Bearing
        axs[1, 1].plot(right_freqs, right_amps)
        axs[1, 1].set_xlim(1, max(right_freqs))
        axs[1, 1].set_title('Right Bearing - Frequency Domain')
        axs[1, 1].set_xlabel('Frequency (Hz)')
        axs[1, 1].set_ylabel('Amplitude (g)')
        axs[1, 1].grid(True)

        # Adjust layout and display the plots
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def run_time_response_imb(self, sampled_params):
        speed = sampled_params['speed']
        imb_vals = [(sampled_params['left_rotor_imb_mag'], sampled_params['left_rotor_imb_phase']), (sampled_params['right_rotor_imb_mag'], sampled_params['right_rotor_imb_phase'])]
        samp_int = self.inputs['sampling_interval']
        sim_samples = self.inputs['simulation_samples']
        ss_samples = self.inputs['steady_state_samples']

        radial_stiffness = sampled_params['rad_stiff']
        bending_stiffness = sampled_params['bend_stiff']
        x_offset = sampled_params['x_misalign']
        y_offset = sampled_params['y_misalign']
        misalignment_angle = sampled_params['misalign_angle']

        # make sure length of imb_vals is equal to number of disks
        if len(imb_vals) != len(self.rotor.disk_elements):
            raise Exception("length of imb_vals must be equal to the number of disks on the rotor.")
        t = np.arange(0,sim_samples+ss_samples) * samp_int
        F = np.zeros((sim_samples+ss_samples, self.rotor.ndof))
        # apply imbalance on each disk
        for i, iv in enumerate(imb_vals):
            mag = iv[0]
            phase = iv[1]
            disk_node = self.rotor.disk_elements[i].n
            # x component
            F[:, 4 * disk_node + 0] = (speed**2) * mag * np.cos(speed * t + phase)
            # y component
            F[:, 4 * disk_node + 1] = (speed**2) * mag * np.sin(speed * t + phase)

        misalignment = self.rotor.run_misalignment(
            coupling="flex",
            dt = samp_int,
            tI = 0,
            tF = (sim_samples + ss_samples) * samp_int,
            kd = radial_stiffness,
            ks = bending_stiffness,
            eCOUPx = x_offset,
            eCOUPy = y_offset,
            misalignment_angle = misalignment_angle,
            TD = 0,
            TL = 0,
            n1 = 0,
            speed = speed,
            unbalance_magnitude = np.array([sampled_params['left_rotor_imb_mag'], sampled_params['right_rotor_imb_mag']]),
            unbalance_phase = np.array([sampled_params['left_rotor_imb_phase'], sampled_params['right_rotor_imb_phase']]),
            mis_type = "parallel",
            print_progress = False
        )

        time_response = misalignment.run_time_response()

        # displacements
        x_disp_0 = time_response.yout[sim_samples:, self.rotor.nodes[self.n_b0]*4 + 0][:-1]
        y_disp_0 = time_response.yout[sim_samples:, self.rotor.nodes[self.n_b0]*4 + 1][:-1]
        x_disp_1 = time_response.yout[sim_samples:, self.rotor.nodes[-1]*4 + 0][:-1]
        y_disp_1 = time_response.yout[sim_samples:, self.rotor.nodes[-1]*4 + 1][:-1]

        # accelerations
        x_acc_0 = self.calculate_acc(x_disp_0, t[sim_samples:])
        y_acc_0 = self.calculate_acc(y_disp_0, t[sim_samples:])
        x_acc_1 = self.calculate_acc(x_disp_1, t[sim_samples:])
        y_acc_1 = self.calculate_acc(y_disp_1, t[sim_samples:])

        # frequency response results
        x_fft_0 = self.compute_fft_with_window(x_acc_0, samp_int, ss_samples)
        y_fft_0 = self.compute_fft_with_window(y_acc_0, samp_int, ss_samples)
        x_fft_1 = self.compute_fft_with_window(x_acc_1, samp_int, ss_samples)
        y_fft_1 = self.compute_fft_with_window(y_acc_1, samp_int, ss_samples)

        # time response results
        x_time_0 = {'time': t[:-2], 'acc': x_acc_0, 'disp': x_disp_0}
        y_time_0 = {'time': t[:-2], 'acc': y_acc_0, 'disp': y_disp_0}
        x_time_1 = {'time': t[:-2], 'acc': x_acc_1, 'disp': x_disp_1}
        y_time_1 = {'time': t[:-2], 'acc': y_acc_1, 'disp': y_disp_1}

        time_results = {'probe_x_0': x_time_0, 'probe_y_0': y_time_0, 'probe_x_1': x_time_1, 'probe_y_1': y_time_1}
        frequency_results = {'probe_x_0': x_fft_0, 'probe_y_0': y_fft_0, 'probe_x_1': x_fft_1, 'probe_y_1': y_fft_1}

        # Plot simulated responses
        # self.plot_sim_responses(time_results, frequency_results)

        return time_results, frequency_results

    def detect_peaks(self, frequencies, amplitudes, height=0.01, prominence=0.005):
        peaks, properties = find_peaks(amplitudes, height=height, prominence=prominence)
        if len(peaks) == 0: # No peaks detected
            return np.array([]), np.array([]) # Return empty arrays
        return frequencies[peaks], amplitudes[peaks] # Return peak frequencies and amplitudes

    def match_peaks(self, real_freqs, pred_freqs, freq_tolerance):
        matched_indices = []
        unmatched_real = []
        unmatched_pred = list(range(len(pred_freqs))) # Initially all predicted peaks are unmatched

        for i, f_real in enumerate(real_freqs):
            distances = np.abs(pred_freqs - f_real)
            closest_idx = np.argmin(distances)

            if distances[closest_idx] <= freq_tolerance:
                matched_indices.append((i, closest_idx))
                unmatched_pred.remove(closest_idx) # Mark this predicted peak as matched
            else:
                unmatched_real.append(i) # No match within tolerance

        return matched_indices, unmatched_real, unmatched_pred

    def calculate_peak_error(self, real_freqs, real_amps, pred_freqs, pred_amps, matched_indices, unmatched_real, unmatched_pred,
                             w_a=0.8, w_f=0.2, freq_tolerance=2.5):
        if len(real_amps) == 0: # No real peaks detected
            logging.warning("No real peaks detected. Returning maximum error.")
            return 1.0 # Return a maximum or default error value

        if len(pred_amps) == 0: # No predicted peaks detected
            logging.warning("No predicted peaks detected. Returning maximum error.")
            return 1.0 # Return a maximum or default error value

        total_error = 0
        max_real_freqs = max(real_freqs)
        max_real_amps = max(real_amps) # Find the largest real amplitude for normalization

        # Error for matched peaks
        for real_idx, pred_idx in matched_indices:
            freq_error = w_f * np.abs(real_freqs[real_idx] - pred_freqs[pred_idx]) / max_real_freqs
            amp_error = w_a * np.abs(real_amps[real_idx] - pred_amps[pred_idx]) / max_real_amps
            total_error += freq_error + amp_error

        # Weighted penalty for unmatched real peaks
        for real_idx in unmatched_real:
            # Find the local maximum in the predicted spectrum within the tolerance range
            local_range = np.abs(pred_freqs - real_freqs[real_idx]) <= freq_tolerance
            local_max_amp = max(pred_amps[local_range], default=0)  # Default to 0 if no points in range
            penalty = (real_amps[real_idx] - local_max_amp) / (real_amps[real_idx] + 1e-6)
            total_error += penalty

        # Fixed penalty for unmatched predicted peaks
        fixed_penalty = 0.5 # Adjust based on the desired penalty for extra predicted peaks
        total_error += fixed_penalty * len(unmatched_pred)

        # Normalize by the number of real peaks
        return total_error / len(real_freqs)

    def peak_based_error(self, real_freqs, real_amps, pred_freqs, pred_amps, freq_tolerance=2.5, w_a=0.8, w_f=0.2):
        # Detect peaks in real and predicted spectra
        real_peaks, real_amplitudes = self.detect_peaks(real_freqs, real_amps)
        pred_peaks, pred_amplitudes = self.detect_peaks(pred_freqs, pred_amps)

        # Match peaks
        matched_indices, unmatched_real, unmatched_pred = self.match_peaks(real_peaks, pred_peaks, freq_tolerance)

        # Calculate error
        return self.calculate_peak_error(real_peaks, real_amplitudes, pred_peaks, pred_amplitudes,
                                    matched_indices, unmatched_real, unmatched_pred,
                                    w_a=w_a, w_f=w_f, freq_tolerance=freq_tolerance)

    def sample_reward(self):
        """
        Sample a reward value from the Beta distribution defined by alpha and beta.
        """
        reward = stats.beta.rvs(self.alpha, self.beta)
        logger.debug(f"Sampled reward for {self.name}: {reward}")
        return reward

    def calibrate_parameters_abc(self, real_data, tolerance=0.1, w1=0.5, w2=0.5, final=False):
        """
        Calibrates model using a peak based error metric.
        """
        tolerance = max(tolerance, 0.05)
        if final:
            sampled_params = self.sample_parameters_final()
        else:
            sampled_params = self.sample_parameters()

        try:
            self.build_rotor(sampled_params)
        except Exception as e:
            logging.error(f"Error in building rotor for model '{self.name}': {e}")
            return 1.0  # Return worst-case peak based error

        try:
            time_results, frequency_results = self.run_time_response_imb(sampled_params)
        except Exception as e:
            logging.error(f"Error during rotor simulation for model '{self.name}': {e}")
            return 1.0  # Return worst-case peak based error

        try:
            # Extract real and predicted data
            left_freqs_real = real_data['frequency']['left']['freq']
            left_magnitude_real = real_data['frequency']['left']['amp']
            right_freqs_real = real_data['frequency']['right']['freq']
            right_magnitude_real = real_data['frequency']['right']['amp']

            left_freqs_pred = frequency_results['probe_x_0']['freq']
            left_magnitude_pred = frequency_results['probe_x_0']['amp']
            right_freqs_pred = frequency_results['probe_x_1']['freq']
            right_magnitude_pred = frequency_results['probe_x_1']['amp']

            # Calculate peak based error for left and right bearings
            left_peak_based_error = self.peak_based_error(
                left_freqs_real, left_magnitude_real, left_freqs_pred, left_magnitude_pred
            )
            right_peak_based_error = self.peak_based_error(
                right_freqs_real, right_magnitude_real, right_freqs_pred, right_magnitude_pred
            )

            avg_peak_based_error = (left_peak_based_error + right_peak_based_error) / 2

        except Exception as e:
            logging.error(f"Error in frequency response comparison for model '{self.name}': {e}")
            return 1.0  # Return worst-case peak based error

        # Step 5: Evaluate based on peak based error threshold
        if avg_peak_based_error < tolerance:  # Lower peak based error is better
            logging.info(f"Accepted parameters for {self.name} with peak based error {avg_peak_based_error:.3f}")
            for param_name, value in sampled_params.items():
                self.accepted_samples[param_name].append(value)
            self.update_beta_distr(success=True, error=avg_peak_based_error, tolerance=tolerance)
        else:
            logging.info(f"Rejected parameters for {self.name} with peak based error {avg_peak_based_error:.3f}")
            self.update_beta_distr(success=False, error=avg_peak_based_error, tolerance=tolerance)

        return avg_peak_based_error

    def update_beta_distr(self, success, error=None, tolerance=None):
        """
        Update the beta distribution based on success or failure of the calibration.
        """
        # Ensure error and tolerance are valid
        if error is None or tolerance is None:
            raise ValueError("Error and tolerance must be provided.")

        # Calculate adjustment factor (bounded between 0 and 1)
        adjustment = max(0, min(1, (tolerance - error) / tolerance))

        if success and self.alpha < 20:
            # Increase alpha only on success
            self.alpha += 1 + adjustment
        elif not success and self.beta < 20:
            # Increase beta on failure
            self.beta += 1 + (1 - adjustment)

        # Prevent alpha and beta from growing too large
        if (self.alpha >= 20 and self.beta >= 20):
            self.alpha /= 10
            self.beta /= 10

        # Log updated beta distribution values
        logging.info(f"Updated beta distribution for {self.name}: alpha={self.alpha:.3f}, beta={self.beta:.3f}")

class DataExtraction:
    def __init__(self, file_name=None, sampling_rate=20000, ss_samples=600000, sim_time = 5):
        self.file_name = file_name
        self.sampling_rate = sampling_rate
        self.ss_samples = ss_samples
        self.sim_time = sim_time

    def volts_2_gs_ch0(self, v):
        p_ch0 = [1.38908943, 0.00211214]
        return v * p_ch0[0] + p_ch0[1]

    def volts_2_gs_ch1(self, v):
        p_ch1 = [1.39154522, 0.00333128]
        return v * p_ch1[0] + p_ch1[1]

    def get_time_signal(self, data_file):
        try:
            with open(data_file, 'r') as f:
                time_signal_full = json.load(f)['value']
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Error loading data: {e}")
            return None, None

        ids_ch0 = np.arange(0, len(time_signal_full), 2)
        ids_ch1 = np.arange(1, len(time_signal_full), 2)

        time_signal0_full = self.volts_2_gs_ch0(np.array([time_signal_full[i] for i in ids_ch0]))
        time_signal1_full = self.volts_2_gs_ch1(np.array([time_signal_full[i] for i in ids_ch1]))

        rand_start = np.random.randint(self.ss_samples - self.sim_time * self.sampling_rate)
        rand_end = rand_start + self.sim_time * self.sampling_rate

        time_signal0 = time_signal0_full[rand_start:rand_end]
        time_signal1 = time_signal1_full[rand_start:rand_end]

        return time_signal0, time_signal1

    def apply_window(self, signal):
        """Applies a Hann window to the signal."""
        window = np.hanning(len(signal)) # Generate a Hann window
        return signal * window, window # Return both windowed signal and the window itself

    def compute_real_fft_with_window(self, signal, sampling_rate, detrend_data=True, window_data = True):
        """Compute the FFT with a Hann window and proper normalization."""
        if detrend_data:
            signal = detrend(signal)

        N = len(signal)

        # Remove DC offset
        signal = signal - np.mean(signal)

        # Apply the Hann window
        if window_data:
            signal, window = self.apply_window(signal)

        # Perform FFT
        fft_values = np.fft.fft(signal)

        # Normalize the FFT using RMS of the window
        rms_window = np.sqrt(np.mean(window**2))
        fft_magnitude = (2 * np.abs(fft_values) / N) / rms_window
        fft_phase = np.angle(fft_values)

        freqs = np.fft.fftfreq(N, d=1 / sampling_rate)

        mask = (freqs >= 5) & (freqs <= 100)

        return freqs[mask], fft_magnitude[mask], fft_phase[mask]

    def plot_real_responses(self, left_time_signal, right_time_signal, left_freqs, left_magnitude, right_freqs, right_magnitude):
                # Plot time-domain and frequency-domain signals
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Time and Frequency Domain Analysis', fontsize=20)

        # Time-domain plots
        axs[0, 0].plot(np.arange(len(left_time_signal)) / self.sampling_rate, left_time_signal)
        axs[0, 0].set_title('Left Bearing - Time Domain')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Acceleration (g)')
        axs[0, 0].grid(True)

        axs[0, 1].plot(np.arange(len(right_time_signal)) / self.sampling_rate, right_time_signal)
        axs[0, 1].set_title('Right Bearing - Time Domain')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Acceleration (g)')
        axs[0, 1].grid(True)

        # Frequency-domain plots
        axs[1, 0].plot(left_freqs, left_magnitude)
        axs[1, 0].set_xlim(1, max(left_freqs))
        axs[1, 0].set_title('Left Bearing - Frequency Domain')
        axs[1, 0].set_xlabel('Frequency (Hz)')
        axs[1, 0].set_ylabel('Amplitude (g)')
        axs[1, 0].grid(True)

        axs[1, 1].plot(right_freqs, right_magnitude)
        axs[1, 1].set_xlim(1, max(right_freqs))
        axs[1, 1].set_title('Right Bearing - Frequency Domain')
        axs[1, 1].set_xlabel('Frequency (Hz)')
        axs[1, 1].set_ylabel('Amplitude (g)')
        axs[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def extract_data(self):
        # Load time-domain signals
        left_time_signal, right_time_signal = self.get_time_signal(self.file_name)
        if left_time_signal is None or right_time_signal is None:
            print("Error: One or both signals could not be loaded.")
            return None

        # Compute FFT for both bearings
        left_freqs, left_magnitude, left_phase = self.compute_real_fft_with_window(left_time_signal, self.sampling_rate)
        right_freqs, right_magnitude, right_phase = self.compute_real_fft_with_window(right_time_signal, self.sampling_rate)

        # # Smooth FFT magnitude data
        # left_freqs, left_magnitude = self.smooth_with_rolling_average(left_freqs, left_magnitude)
        # right_freqs, right_magnitude = self.smooth_with_rolling_average(right_freqs, right_magnitude)

        # Remove DC offset
        left_time_signal = left_time_signal - np.mean(left_time_signal)
        right_time_signal = right_time_signal - np.mean(right_time_signal)

        # Plot real vibration data
        self.plot_real_responses(left_time_signal, right_time_signal, left_freqs, left_magnitude, right_freqs, right_magnitude)

        # Return structured data for use in calibration
        return {
            "time": {"left": left_time_signal, "right": right_time_signal},
            "frequency": {
                "left": {"freq": left_freqs, "amp": left_magnitude, "phase": left_phase},
                "right": {"freq": right_freqs, "amp": right_magnitude, "phase": right_phase},
            }
        }

class RLTrainer:
    def __init__(self, models, file_path, epsilon, num_episodes=500, reward_function=None,
                 initial_tolerance=0.5, decay_rate=0.05, warmup_episodes=20, max_tolerance=1.0,
                 window_size=10, low_success_threshold=0.2, save_file='training_results.xlsx'):
        self.models = models
        self.file_path = file_path
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.reward_function = reward_function or self.default_reward_function
        self.rewards_history = []
        self.initial_tolerance = initial_tolerance
        self.decay_rate = decay_rate
        self.warmup_episodes = warmup_episodes
        self.max_tolerance = max_tolerance
        self.window_size = window_size
        self.low_success_threshold = low_success_threshold
        self.recent_success_rates = deque(maxlen=window_size)
        self.save_file = save_file

    def adaptive_tolerance(self, episode):
        """
        Adjusts the tolerance based on the smoothed average peak based error from recent episodes.
        """
        # Step 1: Warm-up phase
        if episode < self.warmup_episodes:
            return self.initial_tolerance

        # Step 2: Calculate the smoothed average peak based error
        if self.rewards_history:
            smoothed_avg_error = np.mean(self.rewards_history)
        else:
            smoothed_avg_error = self.initial_tolerance # Fallback to initial tolerance if no history exists

        # Step 3: Invert peak based error to calculate tolerance (lower peak based error -> higher tolerance)
        tolerance = max(0.05, min(smoothed_avg_error, 10.0)) # Tolerance inversely proportional to error

        logger.info(f"Episode {episode}: Smoothed avg peak based error = {smoothed_avg_error:.3f}, Tolerance = {tolerance:.3f}")
        return tolerance

    def default_reward_function(self, model):
        return model.sample_reward()

    def select_model(self, episode, epsilon=None):
        """Select a model using epsilon-greedy strategy with decaying epsilon."""
        if epsilon is None:
            epsilon = self.epsilon * (0.99 ** episode)  # Decay epsilon each episode
        if np.random.rand() < epsilon:
            selected_model = np.random.choice(self.models)  # Random exploration
            logger.info(f"Randomly selected model '{selected_model.name}' for exploration with epsilon {epsilon}.")
        else:
            # Exploit: Select model with highest sampled reward
            sampled_rewards = [self.reward_function(model) for model in self.models]
            selected_model = self.models[sampled_rewards.index(max(sampled_rewards))]
            logger.info(f"Selected model '{selected_model.name}' based on highest reward.")
        return selected_model

    def plot_beta_distributions(self, episode=None):
        """Plot Beta distributions for all models."""
        plt.figure(figsize=(15, 5))
        for i, model in enumerate(self.models):
            x = np.linspace(0, 1, 100)
            y = stats.beta.pdf(x, model.alpha, model.beta)
            plt.plot(x, y, label=f'{model.name} (α={model.alpha:.2f}, β={model.beta:.2f})')
        if episode is not None:
            plt.title(f'Beta Distributions at Episode {episode + 1}')
        elif episode is None:
            plt.title(f'Beta Distributions at Training End')
        plt.xlabel('Probability')
        plt.ylabel('Density')
        plt.legend()
        if episode is not None:
            plt.savefig(f'beta_distribution_episode_{episode + 1}.png')
        plt.close

    def train(self):
        """Main training loop using peak based error metric and adaptive tolerance."""
        alpha = 0.4  # EMA smoothing factor

        # Initialize resources monitoring
        process = psutil.Process()
        start_time = time.time()
        initial_cpu = process.cpu_percent(interval=None)
        initial_memory = process.memory_info().rss / (1024 ** 2)
        episode_data = []

        for episode in range(self.num_episodes):
            rl_start_time = time.time()
            model = self.select_model(episode=episode)
            rl_end_time = time.time()
            rl_time = rl_end_time - rl_start_time

            # Extract real data for calibration
            real_data_extractor = DataExtraction(file_name=self.file_path)
            real_data = real_data_extractor.extract_data()

            # Use adaptive tolerance for calibration
            current_tolerance = self.adaptive_tolerance(episode + 1)

            # Calibrate model parameters and calculate the peak based error
            peak_based_error = model.calibrate_parameters_abc(
                real_data=real_data,
                tolerance=current_tolerance
            )
            logger.info(f"Episode {episode + 1}/{self.num_episodes}: Model '{model.name}' calibrated with tolerance {current_tolerance:.3f} and peak based error {peak_based_error:.3f}.")

            current_time = time.time()
            elapsed_time = current_time - start_time
            abc_time = current_time - rl_end_time
            current_cpu = process.cpu_percent(interval=None)
            current_memory = process.memory_info().rss / (1024 ** 2)
            logger.info(
                f"Episode {episode + 1}: Elapsed time = {elapsed_time:.2f} seconds, "
                f"CPU usage = {current_cpu:.2f}%, Memory usage = {current_memory:.2f} MB."
            )

            # Track episode data
            episode_data.append({
                'Episode': episode + 1,
                'Model': model.name,
                'Error': peak_based_error,
                'Elapsed Time (s)': elapsed_time,
                'RL Time (s)': rl_time,
                'Calibration Time (s)': abc_time,
                'CPU Usage': current_cpu,
                'Memory Usage': current_memory
            })

            # Apply EMA smoothing to peak based error
            if self.rewards_history:
                smoothed_value = alpha * peak_based_error + (1 - alpha) * self.rewards_history[-1]
            else:
                smoothed_value = peak_based_error

            # Update rewards history with the smoothed peak based error
            self.rewards_history.append(smoothed_value)

            if len(self.rewards_history) > self.window_size:
                self.rewards_history.pop(0)

            # # Optionally plot distributions every episode
            # if episode % 1 == 0 and episode is not None:
            #     self.plot_beta_distributions(episode=episode)

            # --- Update parameter distributions if any parameter reaches 100 accepted samples ---
            if isinstance(model, RotorModel):
                for param_name, samples in model.accepted_samples.items():
                    if len(samples) >= 100:
                        mean = np.mean(samples)
                        stddev = np.std(samples)

                        # Update the parameter distribution
                        parameter_obj = model.parameters[param_name]  # Assuming parameters are stored in a dictionary
                        parameter_obj.update_parameter_distribution(new_mean=mean, new_stddev=stddev)

                        # Log the update
                        logger.info(f"Updated distribution for parameter '{param_name}' in model '{model.name}' with mean={mean:.3f}, stddev={stddev:.3f}.")

                        # Reset accepted samples for this parameter
                        model.accepted_samples[param_name] = []

            if isinstance(model, SurrogateModel):
                if len(model.accepted_samples) >= 100:
                    logging.info(f"Retraining {model.name} on {len(model.accepted_samples)} new samples.")
                    retraining_data = model.og_training_data + model.accepted_samples
                    model.train(retraining_data)
                    model.accepted_samples = []

        # Log final resource usage
        end_time = time.time()
        final_cpu = process.cpu_percent(interval=None)
        final_memory = process.memory_info().rss / (1024 ** 2)

        # Save episode data to Excel
        df = pd.DataFrame(episode_data)
        df.to_excel(self.save_file, index=False)

        logger.info(f"Training completed in {end_time - start_time:.2f} seconds.")
        logger.info(f"CPU usage: Start={initial_cpu:.2f}%, End={final_cpu:.2f}%.")
        logger.info(f"Memory usage: Start={initial_memory:.2f} MB, End={final_memory:.2f} MB.")

    def evaluate_best_model(self):
        """Evaluate the best model based on its beta distribution."""
        model_rewards = [model.alpha / (model.alpha + model.beta) for model in self.models]
        best_model_index = np.argmax(model_rewards)
        best_model = self.models[best_model_index]

        # Extract real data for calibration
        real_data_extractor = DataExtraction(file_name=self.file_path)
        real_data = real_data_extractor.extract_data()

        # Calibrate model parameters and calculate the peak based error
        peak_based_error = best_model.calibrate_parameters_abc(
            real_data=real_data,
            tolerance=1
        )
        logger.info(f"Best Model: {best_model.name} with reward {model_rewards[best_model_index]:.3f} and peak based error {peak_based_error:.3f}.")

        return best_model

    def plot_posterior_distribution(self, samples, param_name, model_name, ax):
        """
        Plots the posterior distribution for a given parameter on the provided subplot axis.
        """
        mean = np.mean(samples)
        stddev = np.std(samples)
        median = np.median(samples)
        skewness = stats.skew(samples)
        kurtosis_value = stats.kurtosis(samples)

        logger.info(f"{model_name} {param_name} mean value: {mean:.3f}")

        # Plot histogram of accepted samples
        ax.hist(samples, bins=20, density=True, alpha=0.5, color='g', label='Accepted Samples')
        sns.kdeplot(samples, color='r', label='Posterior Distribution', linewidth=2, ax=ax)

        # Title and statistics
        ax.set_title(f'{model_name}: {param_name}\nMean: {mean:.3f}, Stddev: {stddev:.3f}, Median: {median:.3f}', fontsize=10)
        stats_text = (f'Mean: {mean:.3f}\n'
                      f'Stddev: {stddev:.3f}\n'
                      f'Median: {median:.3f}\n'
                      f'Skewness: {skewness:.3f}\n'
                      f'Kurtosis: {kurtosis_value:.3f}')
        ax.text(0.98, 0.85, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.6))

        # Labels
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Density')
        ax.legend()

    def log_and_plot_posteriors(self):
        """
        Logs and plots posterior distributions for all models:
        - Damping and stiffness parameters are plotted in their own grid.
        - Remaining parameters are plotted in a separate grid.
        """
        for model in self.models:
            if isinstance(model, RotorModel):
                logger.info(f"Posterior distributions for {model.name}:")

                # Classify parameters
                damping_stiffness_params = [
                    p for p in model.accepted_samples.keys()
                    if 'kxx' in p or 'kxy' in p or 'kyx' in p or 'kyy' in p or 'cxx' in p or 'cxy' in p or 'cyx' in p or 'cyy' in p
                ]
                other_params = [
                    p for p in model.accepted_samples.keys()
                    if p not in damping_stiffness_params and model.parameters[p].shape != 'constant'
                ]

                # --- Plot grid for damping and stiffness parameters ---
                if damping_stiffness_params:
                    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
                    axes = axes.flatten()

                    for i, param_name in enumerate(damping_stiffness_params):
                        samples = model.accepted_samples.get(param_name, [])
                        if samples:
                            logger.info(f"Plotting posterior distribution for {param_name} in model {model.name}.")
                            self.plot_posterior_distribution(samples, param_name, model.name, ax=axes[i])

                    # Hide unused subplots in the grid
                    for j in range(i + 1, len(axes)):
                        axes[j].axis('off')

                    plt.tight_layout()
                    plt.show()

                # --- Plot grid for other parameters ---
                if other_params:
                    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
                    axes = axes.flatten()

                    for i, param_name in enumerate(other_params):
                        samples = model.accepted_samples.get(param_name, [])
                        if samples:
                            logger.info(f"Plotting posterior distribution for {param_name} in model {model.name}.")
                            self.plot_posterior_distribution(samples, param_name, model.name, ax=axes[i])

                    # Hide unused subplots in the grid
                    for j in range(i + 1, len(axes)):
                        axes[j].axis('off')

                    plt.tight_layout()
                    plt.show()

                # --- Plot for speed parameter separately, if it exists ---
                speed_samples = model.accepted_samples.get('speed', [])
                if speed_samples:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    logger.info(f"Plotting posterior distribution for speed in model {model.name}.")
                    self.plot_posterior_distribution(speed_samples, 'speed', model.name, ax=ax)
                    plt.show()

    def plot_debugging_info(self):
        # Plot Beta Distributions
        # self.plot_beta_distributions()

        # Plot Posterior Distributions
        self.log_and_plot_posteriors()

        # Display all plots at the end
        plt.show()

    def run(self):
        """Run the training and return the best model."""
        logger.info("Starting RLTrainer run.")
        total_start_time = time.time()

        best_model = None
        try:
            self.train()
            best_model = self.evaluate_best_model()
            self.plot_debugging_info()
        finally:
            total_end_time = time.time()
            logger.info(f"RLTrainer run completed in {total_end_time - total_start_time:.2f} seconds.")

        return best_model

class SurrogateModel:
    def __init__(self, name, **kwargs):
        """Initialize storage for probability distributions and Bayesian model."""
        self.name = name
        self.alpha = kwargs.get('alpha', 2)
        self.beta = kwargs.get('beta', 2)
        self.betadistr = [self.alpha, self.beta]
        self.peak_count_probs = {"left": defaultdict(int), "right": defaultdict(int)}
        self.peak_distributions = {"left": {}, "right": {}}
        self.peak_means = {"left": {}, "right": {}}
        self.bayesian_model = None
        self.inference = None
        self.og_training_data = []
        self.accepted_samples = []

    def train(self, training_data):
        """Train probability distributions based on training data."""
        peak_counts = {"left": [], "right": []}
        peak_values = {"left": {1: [], 2: [], 3: []}, "right": {1: [], 2: [], 3: []}}

        for data in training_data:
            left_peaks, right_peaks = data["left"], data["right"]
            left_peaks, right_peaks = self._pad_missing_peaks(left_peaks, right_peaks)

            peak_counts["left"].append(len(left_peaks))
            peak_counts["right"].append(len(right_peaks))

            for i, peak in enumerate(left_peaks[:3]):
                peak_values["left"][i + 1].append(tuple(peak))
            for i, peak in enumerate(right_peaks[:3]):
                peak_values["right"][i + 1].append(tuple(peak))

        # Compute peak count probabilities
        for side in ["left", "right"]:
            unique_counts, counts = np.unique(peak_counts[side], return_counts=True)
            total = sum(counts)
            self.peak_count_probs[side] = {int(k): v / total for k, v in zip(unique_counts, counts)}

        # Fit probability distributions for frequencies and magnitudes
        for side in ["left", "right"]:
            for peak_num in range(1, 4):
                if peak_values[side][peak_num]:
                    freqs, mags = zip(*peak_values[side][peak_num])

                    # Apply filtering to remove outliers (kept within 3 std deviations)
                    freqs_filtered = self._filter_outliers(np.array(freqs))
                    mags_filtered = self._filter_outliers(np.array(mags))

                    if len(freqs_filtered) > 0 and len(mags_filtered) > 0:
                        self.peak_distributions[side][peak_num] = {
                            "freq": stats.norm.fit(freqs),
                            "mag": stats.norm.fit(mags),
                        }
                        # Store mean values for padding
                        self.peak_means[side][peak_num] = (np.mean(freqs_filtered), np.mean(mags_filtered))

        # Train Bayesian Network
        self._train_bayesian_network(training_data)

    def _pad_missing_peaks(self, left_peaks, right_peaks):
        """Pads missing peak values using stored means from training data."""
        for side, peaks in zip(["left", "right"], [left_peaks, right_peaks]):
            while len(peaks) < 3:
                peak_num = len(peaks) + 1
                if peak_num in self.peak_means[side]:
                    peaks.append(self.peak_means[side][peak_num])
                else:
                    peaks.append((0, 0))
        return left_peaks[:3], right_peaks[:3]
    
    def _filter_outliers(self, values, threshold=3.0):
        """Removes values beyond the given threshold of standard deviations."""
        mean = np.mean(values)
        std_dev = np.std(values)
        filtered_values = [v for v in values if abs(v - mean) <= threshold * std_dev]
        return np.array(filtered_values)

    def _train_bayesian_network(self, training_data):
        """Train a Bayesian Network to model dependencies between peak frequencies."""
        df = []

        for data in training_data:
            left_peaks, right_peaks = self._pad_missing_peaks(data["left"], data["right"])
            row = {
                "left_freq_1": left_peaks[0][0],
                "left_freq_2": left_peaks[1][0],
                "left_freq_3": left_peaks[2][0],
                "right_freq_1": right_peaks[0][0],
                "right_freq_2": right_peaks[1][0],
                "right_freq_3": right_peaks[2][0],
            }
            df.append(row)

        df = pd.DataFrame(df)
        df = df.fillna(df.mean())

        model = BayesianNetwork([
            ("left_freq_1", "left_freq_2"),
            ("left_freq_2", "left_freq_3"),
            ("right_freq_1", "right_freq_2"),
            ("right_freq_2", "right_freq_3"),
        ])

        model.fit(df, estimator=BayesianEstimator)
        self.bayesian_model = model
        self.inference = VariableElimination(model)

    def sample_reward(self):
        """
        Sample a reward value from the Beta distribution defined by alpha and beta.
        """
        reward = stats.beta.rvs(self.alpha, self.beta)
        logger.debug(f"Sampled reward for {self.name}: {reward}")
        return reward

    def generate_prediction(self):
        """Generate a new prediction using Bayesian inference and probability distributions."""
        prediction = {"left": [], "right": []}
        
        for side in ["left", "right"]:
            peak_count = random.choices(
                list(self.peak_count_probs[side].keys()),
                weights=list(self.peak_count_probs[side].values()),
                k=1
            )[0]

            peak_freq_vars = [f"{side}_freq_{i}" for i in range(1, peak_count + 1)]
            evidence = {}

            for i, var in enumerate(peak_freq_vars):
                # If this is the first peak, sample from the fitted Gaussian
                if i == 0:
                    params = self.peak_distributions[side].get(i + 1, {}).get("freq", (0, 1))
                    freq = np.random.normal(params[0], params[1])
                    evidence[var] = freq
                else:
                    # Use inference to conditionally sample this frequency given previous ones
                    try:
                        result = self.inference.query(
                            variables=[var],
                            evidence=evidence,
                            show_progress=False
                        )
                        freq_vals = list(map(float, result.state_names[var]))
                        freq_probs = result.values / result.values.sum()
                        freq = np.random.choice(freq_vals, p=freq_probs)
                        evidence[var] = freq
                    except Exception as e:
                        logging.warning(f"Bayesian inference failed for {var}: {e}")
                        # Fall back to normal distribution sampling
                        params = self.peak_distributions[side].get(i + 1, {}).get("freq", (0, 1))
                        freq = np.random.normal(params[0], params[1])
                        evidence[var] = freq

                # Now sample magnitude from Gaussian
                mag_params = self.peak_distributions[side].get(i + 1, {}).get("mag", (0, 1))
                mag = np.random.normal(mag_params[0], mag_params[1])

                prediction[side].append([evidence[var], mag])

        return prediction

    def detect_peaks(self, frequencies, amplitudes, height=0.01, prominence=0.005):
        """Detect peaks in frequency domain vibration data."""
        peaks, _ = find_peaks(amplitudes, height=height, prominence=prominence)
        if len(peaks) == 0:
            return np.array([]), np.array([])
        return frequencies[peaks], amplitudes[peaks]

    def match_peaks(self, real_freqs, pred_freqs, freq_tolerance):
        """Finds the best matching predicted peaks for real peaks within a given tolerance."""
        matched_indices = []
        unmatched_real = []
        unmatched_pred = list(range(len(pred_freqs)))  

        for i, f_real in enumerate(real_freqs):
            distances = np.abs(pred_freqs - f_real)
            closest_idx = np.argmin(distances)

            if distances[closest_idx] <= freq_tolerance:
                matched_indices.append((i, closest_idx))
                try:
                    unmatched_pred.remove(closest_idx)
                except ValueError:
                    pass  
            else:
                unmatched_real.append(i)  

        return matched_indices, unmatched_real, unmatched_pred

    def calculate_peak_error(self, real_freqs, real_amps, pred_freqs, pred_amps, matched_indices, unmatched_real, unmatched_pred,
                             w_a=0.8, w_f=0.2, freq_tolerance=2.5):
        if len(real_amps) == 0: # No real peaks detected
            logging.warning("No real peaks detected. Returning maximum error.")
            return 1.0 # Return a maximum or default error value

        if len(pred_amps) == 0: # No predicted peaks detected
            logging.warning("No predicted peaks detected. Returning maximum error.")
            return 1.0 # Return a maximum or default error value

        total_error = 0
        max_real_freqs = max(real_freqs)
        max_real_amps = max(real_amps) # Find the largest real amplitude for normalization

        # Error for matched peaks
        for real_idx, pred_idx in matched_indices:
            freq_error = w_f * np.abs(real_freqs[real_idx] - pred_freqs[pred_idx]) / max_real_freqs
            amp_error = w_a * np.abs(real_amps[real_idx] - pred_amps[pred_idx]) / max_real_amps
            total_error += freq_error + amp_error

        # Weighted penalty for unmatched real peaks
        for real_idx in unmatched_real:
            # Find the local maximum in the predicted spectrum within the tolerance range
            local_range = np.abs(pred_freqs - real_freqs[real_idx]) <= freq_tolerance
            local_max_amp = max(pred_amps[local_range], default=0)  # Default to 0 if no points in range
            penalty = (real_amps[real_idx] - local_max_amp) / (real_amps[real_idx] + 1e-6)
            total_error += penalty

        # Fixed penalty for unmatched predicted peaks
        fixed_penalty = 0.5 # Adjust based on the desired penalty for extra predicted peaks
        total_error += fixed_penalty * len(unmatched_pred)

        # Normalize by the number of real peaks
        return total_error / len(real_freqs)

    def peak_based_error(self, real_freqs, real_amps, pred_freqs, pred_amps, freq_tolerance=2.5, w_a=0.8, w_f=0.2):
        # Detect peaks in real and predicted spectra
        real_peaks, real_amplitudes = self.detect_peaks(real_freqs, real_amps)
        pred_peaks, pred_amplitudes = np.array(pred_freqs), np.array(pred_amps)

        # Match peaks
        matched_indices, unmatched_real, unmatched_pred = self.match_peaks(real_peaks, pred_peaks, freq_tolerance)

        # Calculate error
        return self.calculate_peak_error(real_peaks, real_amplitudes, pred_peaks, pred_amplitudes,
                                    matched_indices, unmatched_real, unmatched_pred,
                                    w_a=w_a, w_f=w_f, freq_tolerance=freq_tolerance)

    # === Convert Peaks to Nested List Format ===
    def return_nn_data(self, frequencies, amplitudes):
        """
        Converts detected peaks into a list of [frequency, amplitude] pairs.
        """
        return [[frequencies[i], amplitudes[i]] for i in range(len(frequencies))]

    def calibrate_parameters_abc(self, real_data, tolerance=0.1):
        """Calibrates the surrogate model using peak-based error metrics."""
        tolerance = max(tolerance, 0.05)

        try:
            surrogate_prediction = self.generate_prediction()
            print("Surrogate Prediction:", surrogate_prediction)
            left_freqs_pred, left_magnitude_pred = zip(*surrogate_prediction["left"]) if surrogate_prediction["left"] else ([], [])
            right_freqs_pred, right_magnitude_pred = zip(*surrogate_prediction["right"]) if surrogate_prediction["right"] else ([], [])

            left_freqs_real, left_magnitude_real = real_data['frequency']['left']['freq'], real_data['frequency']['left']['amp']
            right_freqs_real, right_magnitude_real = real_data['frequency']['right']['freq'], real_data['frequency']['right']['amp']

            avg_peak_based_error = (self.peak_based_error(left_freqs_real, left_magnitude_real, left_freqs_pred, left_magnitude_pred) + 
                                    self.peak_based_error(right_freqs_real, right_magnitude_real, right_freqs_pred, right_magnitude_pred)) / 2            

        except Exception as e:
            logging.error(f"Error in frequency response comparison for model '{self.name}': {e}")
            return 1.0  # Return worst-case peak based error

        # Step 5: Evaluate based on peak based error threshold
        if avg_peak_based_error < tolerance:  # Lower peak based error is better
            logging.info(f"Accepted parameters for {self.name} with peak based error {avg_peak_based_error:.3f}")
            
            # Detect peaks for left and right side
            left_real_peaks, left_real_amplitudes = self.detect_peaks(left_freqs_real, left_magnitude_real)
            right_real_peaks, right_real_amplitudes = self.detect_peaks(right_freqs_real, right_magnitude_real)

            # Convert detected peaks to list format
            left_nn_data = self.return_nn_data(left_real_peaks, left_real_amplitudes)
            right_nn_data = self.return_nn_data(right_real_peaks, right_real_amplitudes)

            self.accepted_samples.append({"left": left_nn_data, "right": right_nn_data})

            self.update_beta_distr(success=True, error=avg_peak_based_error, tolerance=tolerance)
        else:
            logging.info(f"Rejected parameters for {self.name} with peak based error {avg_peak_based_error:.3f}")
            self.update_beta_distr(success=False, error=avg_peak_based_error, tolerance=tolerance)

        return avg_peak_based_error
    
    def update_beta_distr(self, success, error=None, tolerance=None):
        """
        Update the beta distribution based on success or failure of the calibration.
        """
        # Ensure error and tolerance are valid
        if error is None or tolerance is None:
            raise ValueError("Error and tolerance must be provided.")

        # Calculate adjustment factor (bounded between 0 and 1)
        adjustment = max(0, min(1, (tolerance - error) / tolerance))

        if success and self.alpha < 20:
            # Increase alpha only on success
            self.alpha += 1 + adjustment
        elif not success and self.beta < 20:
            # Increase beta on failure
            self.beta += 1 + (1 - adjustment)

        # Prevent alpha and beta from growing too large
        if (self.alpha >= 20 and self.beta >= 20):
            self.alpha /= 10
            self.beta /= 10

        # Log updated beta distribution values
        logging.info(f"Updated beta distribution for {self.name}: alpha={self.alpha:.3f}, beta={self.beta:.3f}")


# === Detect Peaks Function ===
def detect_peaks(frequencies, amplitudes, height=0.01, prominence=0.005):
    """
    Detects peaks in frequency domain vibration data.
    Returns peak frequencies and corresponding amplitudes.
    """
    peaks, _ = find_peaks(amplitudes, height=height, prominence=prominence)
    if len(peaks) == 0:  # No peaks detected
        return np.array([]), np.array([])
    return frequencies[peaks], amplitudes[peaks]

# === Convert Peaks to Nested List Format ===
def return_nn_data(frequencies, amplitudes):
    """
    Converts detected peaks into a list of [frequency, amplitude] pairs.
    """
    return [[frequencies[i], amplitudes[i]] for i in range(len(frequencies))]

# === Function to Generate Multiple Training Data Points ===
def generate_training_data(num_samples, file_name):
    """
    Generates a user-defined number of training data points in the required format.
    Each iteration extracts peaks and stores them in the correct structure.
    """
    training_data = []

    for _ in range(num_samples):
        # === Load Real Data ===
        real_data_extractor = DataExtraction(file_name=file_name)
        real_data = real_data_extractor.extract_data()

        left_freqs_real = real_data['frequency']['left']['freq']
        left_magnitude_real = real_data['frequency']['left']['amp']
        right_freqs_real = real_data['frequency']['right']['freq']
        right_magnitude_real = real_data['frequency']['right']['amp']

        # Detect peaks for left and right side
        left_real_peaks, left_real_amplitudes = detect_peaks(left_freqs_real, left_magnitude_real)
        right_real_peaks, right_real_amplitudes = detect_peaks(right_freqs_real, right_magnitude_real)

        # Convert detected peaks to list format
        left_nn_data = return_nn_data(left_real_peaks, left_real_amplitudes)
        right_nn_data = return_nn_data(right_real_peaks, right_real_amplitudes)

        # Append formatted training sample
        training_data.append({"left": left_nn_data, "right": right_nn_data})

    return training_data

# === Example: Generate 100 Training Data Points ===
num_samples = 1
file_name = 'IOC_20hz_20khz_30s_2.json'
formatted_training_data = generate_training_data(num_samples, file_name)

# Print an example training data point
print("IOC Training Data Point:", formatted_training_data[0])

IOC_surrogate = SurrogateModel('IOC_surrogate')
IOC_surrogate.og_training_data = formatted_training_data
IOC_surrogate.train(formatted_training_data)

# === Example: Generate 100 Training Data Points ===
num_samples = 1
file_name = '5gimb_20hz_20khz_30s_2.json'
formatted_training_data = generate_training_data(num_samples, file_name)

# Print an example training data point
print("5gimb Training Data Point:", formatted_training_data[0])

imb_surrogate_5g = SurrogateModel('5gimb_surrogate')
imb_surrogate_5g.og_training_data = formatted_training_data
imb_surrogate_5g.train(formatted_training_data)

# === Example: Generate 100 Training Data Points ===
num_samples = 1
file_name = '10gimb_20hz_20khz_30s_2.json'
formatted_training_data = generate_training_data(num_samples, file_name)

# Print an example training data point
print("10gimb Training Data Point:", formatted_training_data[0])

imb_surrogate_10g = SurrogateModel('10gimb_surrogate')
imb_surrogate_10g.og_training_data = formatted_training_data
imb_surrogate_10g.train(formatted_training_data)

# === Example: Generate 100 Training Data Points ===
num_samples = 1
file_name = '15gimb_20hz_20khz_30s_2.json'
formatted_training_data = generate_training_data(num_samples, file_name)

# Print an example training data point
print("15gimb Training Data Point:", formatted_training_data[0])

imb_surrogate_15g = SurrogateModel('15gimb_surrogate')
imb_surrogate_15g.og_training_data = formatted_training_data
imb_surrogate_15g.train(formatted_training_data)

# === Example: Generate 100 Training Data Points ===
num_samples = 1
file_name = '20gimb_20hz_20khz_30s_2.json'
formatted_training_data = generate_training_data(num_samples, file_name)

# Print an example training data point
print("20gimb Training Data Point:", formatted_training_data[0])

imb_surrogate_20g = SurrogateModel('20gimb_surrogate')
imb_surrogate_20g.og_training_data = formatted_training_data
imb_surrogate_20g.train(formatted_training_data)

# === Example: Generate 100 Training Data Points ===
num_samples = 1
file_name = '25gimb_20hz_20khz_30s_2.json'
formatted_training_data = generate_training_data(num_samples, file_name)

# Print an example training data point
print("25gimb Training Data Point:", formatted_training_data[0])

imb_surrogate_25g = SurrogateModel('25gimb_surrogate')
imb_surrogate_25g.og_training_data = formatted_training_data
imb_surrogate_25g.train(formatted_training_data)

trainer = RLTrainer([imb_surrogate_25g], file_path='25gimb_20hz_20khz_30s_1.json', epsilon=0.45, num_episodes=10, initial_tolerance=0.5, decay_rate=0.005, warmup_episodes=1, max_tolerance=1.0, save_file='test.xlsx')
trainer.run()