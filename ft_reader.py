import serial
import time
from collections import deque
import logging
import numpy as np
import scipy

B_PRIME_MAT = np.array([
    [-8, 62, -59, 4, -912, 914],
    [48, -25, -35, -1058, 528, 529],
    [800, 800, 800, 0, 0, 0],
    [-5, 478, -468, -128, 67, -56],
    [550, -268, -278, -5, -107, 116],
    [14, 16, 19, -288, -288, -289]
])


class FTReader:

    def __init__(self, serial_port, baud_rate):
        super().__init__()
        self.serial_port = serial.Serial(serial_port, baud_rate, timeout=0.1)
        self.serial_port.reset_input_buffer()
        self.serial_port.reset_output_buffer()
        self.calibration_values = [0] * 6  # Calibration values for the sensor

        # Buffer for storing the last 200 data points
        self.data_buffer = deque(maxlen=200)
        # Timing for interval measurement
        self.last_update_time = time.time()  # Initialize the last update time

    def get_filtered_latest(self):
        hist = np.vstack(self.data_buffer).T

        # hist -= np.mean(hist, axis=1, keepdims=True)

        b, a = scipy.signal.butter(10, 1/40)
        hist_filtered = scipy.signal.filtfilt(b, a, hist)

        # hist_filtered -= np.mean(hist_filtered, axis=1, keepdims=True)

        return hist_filtered.T[-1]

    def get_latest(self):
        hist = np.vstack(self.data_buffer).T
        return hist.T[-1]

    def warmed_up(self) -> bool:
        return (len(self.data_buffer) > 20)

    def set_zero(self):
        logging.debug(self.data_buffer)
        self.calibration_values = np.mean(self.data_buffer, axis=0)
        logging.debug(self.calibration_values)

    def run_thread(self):
        '''
        channel order:
        ch0 - G1 side
        ch1 - G2 up/down
        ch2 - G2 side
        ch3 - G3 up/down
        ch4 - G3 side
        ch5 - G1 up/down
        '''
        SG_ORDER = [6, 2, 4, 1, 3, 5]
        interval_sum = 0.0  # Sum of intervals measured
        interval_count = 0  # Count of intervals measured
        last_print_time = time.time()  # Last time the average was printed

        channel_buffer = [0, 0, 0, 0, 0, 0]
        channel_count = 0

        self.running = True
        while self.running:
            if self.serial_port.inWaiting() > 0:
                current_time = time.time()
                actual_interval = (
                    current_time - self.last_update_time) * 1000  # ms
                self.last_update_time = current_time

                # Accumulate the sum and count of intervals
                interval_sum += actual_interval
                interval_count += 1

                try:
                    line = self.serial_port.readline().decode().strip()
                    if line:  # If the line is not empty
                        try:
                            if not line.startswith("ch"):
                                continue

                            channel, value = line.split("=")
                            channel = int(channel[2:])
                            value = float(value)

                            channel_buffer[channel - 1] = value
                            channel_count += 1

                            if channel_count == 6:

                                sensor_values = channel_buffer.copy()
                                channel_buffer = [0] * 6
                                channel_count = 0

                                # Initialize a list with the same length as sensor_values filled
                                # with zeros
                                reordered_values = [0] * len(sensor_values)

                                # Reorder the sensor_values according to SG_ORDER
                                # SG_ORDER is treated as the target position for each corresponding
                                # value
                                for i in range(6):
                                    reordered_values[i] = sensor_values[SG_ORDER[i]-1]

                                # Apply calibration values to the reordered values
                                reordered_values = [
                                    reordered_values[i] -
                                    self.calibration_values[i] for i in range(6)]

                                # self.data_buffer.append(
                                #     np.dot(B_PRIME_MAT, reordered_values))

                                self.data_buffer.append(reordered_values)

                        except (ValueError, IndexError) as e:
                            # If conversion fails, it's not a data line
                            logging.info("Message received: %s", line)
                except ValueError:
                    # This catches errors not related to parsing floats,
                    # e.g., if the line could not be decoded from UTF-8
                    logging.debug("Failed to decode serial data")

                # Check if a second has passed since the last print
                if current_time - last_print_time >= 1.0:
                    if interval_count > 0:
                        average_interval = interval_sum / interval_count
                        logging.debug(
                            "Average serial interval: %.2f ms", average_interval)
                    else:
                        logging.debug("No data received in the last second.")

                    # Reset for the next second
                    last_print_time = current_time
                    interval_sum = 0.0
                    interval_count = 0
