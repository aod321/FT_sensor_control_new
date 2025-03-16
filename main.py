import fibre
import ref_tool
import serial
import time
from collections import deque
import threading
import logging
import signal
import numpy as np
import scipy
import keyboard


B_PRIME_MAT = np.array([
    [-8, 62, -59, 4, -912, 914],
    [48, -25, -35, -1058, 528, 529],
    [600, 600, 600, 0, 0, 0],
    [-5, 478, -468, -128, 67, -56],
    [550, -268, -278, -5, -107, 116],
    [14, 16, 19, -288, -288, -289]
])


def make_bar(data: np.ndarray, width: int = 50, scale: int = 1e-6) -> None:
    bar_length = min(abs(int(data * scale)), width)

    if data >= 0:
        disp = f"{' ' * width}[{data:.1f}\t]{'#' * bar_length}{' ' * (width-bar_length)}"
    else:
        disp = f"{' ' * (width-bar_length)}{'#' * bar_length}[{data}\t]{' ' * (width)}"
    return disp


def print_bars(data: np.ndarray, width: int = 50, scale: int = 1e-6) -> None:
    """ Visualize wrench """
    print("\033[H\033[J", end="")  # Clear the screen
    for i in range(len(data)):
        print(f"{i}: {make_bar(data[i], width, scale)}")
    print(f"{str(data)}")


def skew_symmetric(v: np.ndarray[3]) -> np.ndarray[3, 3]:
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


class DummyRobot:
    def __init__(self, real_device=False, serial_id=None):
        self.real_device = real_device
        if real_device:
            logging.info("Connecting to the robot...")
            if serial_id is not None:
                self.arm = ref_tool.find_any(serial_number=serial_id)
            else:
                self.arm = ref_tool.find_any()
            logging.info("Connected to the robot.")
            self.arm.robot.set_enable(True)
            self.arm.robot.homing()

        self.xyz_control_gain = 1e-9
        self.rpy_control_gain = 1e-9

        self.last_pose = self.read_eef_pose()
        self.last_target_pose = self.last_pose

        self.ignored_traget_count = 0

    def home(self):
        self.arm.robot.homing()

    def read_eef_pose(self) -> np.ndarray[6]:
        """ EEF pose is a 6D vector [x, y, z, roll, pitch, yaw] """
        if self.real_device:
            self.arm.robot.eef_pose.update_pose_6D()
            x, y, z, a, b, c = self.arm.robot.eef_pose.x, self.arm.robot.eef_pose.y, self.arm.robot.eef_pose.z, self.arm.robot.eef_pose.a, self.arm.robot.eef_pose.b, self.arm.robot.eef_pose.c
            self.last_pose = np.array([x, y, z, a, b, c])
            return np.array([x, y, z, a, b, c])
        else:
            return np.array([1, 1, 1, 0, 0, 1])

    def set_target_pose(self, target_pose: np.ndarray[6]) -> None:
        """ Set the target pose of the robot """
        # prevent setting a target that is too far from last target or current pose
        if self.last_target_pose is not None and self.last_pose is not None and self.ignored_traget_count < 5:
            if np.any(np.abs(target_pose - self.last_target_pose) > 5) or np.any(np.abs(target_pose - self.last_pose) > 5):
                logging.error(
                    "Target pose too far from last target, ignoring("+str(target_pose)+")")
                self.ignored_traget_count += 1
                return

        self.last_target_pose = target_pose
        self.ignored_traget_count = 0
        logging.info(
            f"Setting target pose to {str(target_pose[:3])}, {target_pose[3:]}")
        if self.real_device:
            self.arm.robot.move_l(
                target_pose[0], target_pose[1], target_pose[2], target_pose[3], target_pose[4], target_pose[5])
            logging.info(f"Set target pose done")

    def update_pose_with_wrench(self, wrench: np.ndarray[6]) -> np.ndarray[6]:
        """ Update the pose of the robot with the given wrench """

        logging.info(f"Updating pose with wrench {str(wrench)} ")

        # ignore wrench if it's too small
        if np.linalg.norm(wrench) < 5e-7:
            logging.info("Wrench too small, ignoring")
            return self.last_pose

        """
        1. 从机械臂得到当前末端 (ee) 相对在base下的位置xyz (p) 和旋转矩阵(R)
        2. 用 p 和 R 得到当前的 adjoint map matrix (Ad)
            p_cross= skew_symmetric(p)
            Ad = np.block([
                [R, np.zeros((3,3))],
                [p_cross @ R, R]
            ])
        3. 从传感器读到当前的力和力矩 (w_ee), 左乘 Ad^T 得到base下的力和力矩 (w_base)
        4. w_base 拆分为 force_base 和 torque_base, 取一个小的比例系数 k, 那么
            target_xyz = current_xyz + k * force_base,
            target_R = curret_R @ scipy.linalg.expm(skew_symmetric(k * torque_base))
        之后把 R 转换为 Euler 角度发送给机械臂执行（或者直接在上位机计算IK）
        """

        eef_pose = self.read_eef_pose()
        logging.info(f"Current EEF pose:{str(eef_pose)}")

        p = eef_pose[:3]
        euler_angles = eef_pose[3:]
        R = scipy.spatial.transform.Rotation.from_euler(
            "xyz", euler_angles, degrees=True).as_matrix()

        p_cross = skew_symmetric(p)
        Ad = np.block([
            [R, np.zeros((3, 3))],
            [p_cross @ R, R]
        ])
        w_ee = np.array(wrench)
        w_base = Ad.T @ w_ee

        force_base = w_base[:3]
        torque_base = w_base[3:]
        target_xyz = p + self.xyz_control_gain * force_base
        target_R = R @ scipy.linalg.expm(
            skew_symmetric(self.rpy_control_gain * torque_base))
        target_euler_angles = scipy.spatial.transform.Rotation.from_matrix(
            target_R).as_euler("xyz", degrees=True)

        target_pose = np.concatenate([target_xyz, target_euler_angles])
        self.set_target_pose(target_pose)
        return target_pose


class FTReader:

    def __init__(self, serial_port, baud_rate):
        super().__init__()
        self.serial_port = serial.Serial(serial_port, baud_rate, timeout=0.1)
        self.serial_port.reset_input_buffer()
        self.serial_port.reset_output_buffer()
        self.calibration_values = [0] * 6  # Calibration values for the sensor

        # Buffer for storing the last 200 data points
        self.data_buffer = deque(maxlen=50)
        # Timing for interval measurement
        self.last_update_time = time.time()  # Initialize the last update time

    def get_filtered_latest(self):
        hist = np.vstack(self.data_buffer).T

        hist -= np.mean(hist, axis=1, keepdims=True)

        b, a = scipy.signal.butter(4, 0.14)
        hist_filtered = scipy.signal.filtfilt(b, a, hist)

        # hist_filtered -= np.mean(hist_filtered, axis=1, keepdims=True)

        return hist_filtered.T[-1]

    def warmed_up(self) -> bool:
        return (len(self.data_buffer) > 20)

    def set_zero(self):
        self.serial_port.write(b"t\n")

    def run_thread(self):
        SG_ORDER = [1, 6, 5, 4, 3, 2]
        interval_sum = 0.0  # Sum of intervals measured
        interval_count = 0  # Count of intervals measured
        last_print_time = time.time()  # Last time the average was printed

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
                            # Attempt to parse the line as a list of floats
                            sensor_values = [float(val)
                                             for val in line.split(",")[:6]]

                            # Initialize a list with the same length as sensor_values filled
                            # with zeros
                            reordered_values = [0] * len(sensor_values)

                            # Reorder the sensor_values according to SG_ORDER
                            # SG_ORDER is treated as the target position for each corresponding
                            # value
                            for i, order in enumerate(SG_ORDER):
                                reordered_values[order - 1] = sensor_values[i]

                            # Apply calibration values to the reordered values
                            reordered_values = [
                                reordered_values[i] -
                                self.calibration_values[i] for i in range(6)]

                            self.data_buffer.append(
                                np.dot(B_PRIME_MAT, reordered_values))
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


class FakeFTReader:
    """ Emulate the FT sensor """

    def __init__(self):
        self.data = np.zeros(6)
        self.keys_add = {
            'q': 0,
            'w': 1,
            'e': 2,
            'r': 3,
            't': 4,
            'y': 5
        }
        self.keys_sub = {
            'a': 0,
            's': 1,
            'd': 2,
            'f': 3,
            'g': 4,
            'h': 5
        }
        self.running = True
        self.setup_key_hooks()

    def setup_key_hooks(self):
        keyboard.on_press(self.on_key_event)

    def on_key_event(self, event):
        key = event.name
        if key in self.keys_add:
            self.data[self.keys_add[key]] += 1
        elif key in self.keys_sub:
            self.data[self.keys_sub[key]] -= 1

    def get_filtered_latest(self):
        return self.data * 1e8

    def warmed_up(self):
        return True

    def run_thread(self):
        while self.running:
            time.sleep(0.01)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    robot = DummyRobot(real_device=True)
    time.sleep(1)
    print(robot.read_eef_pose())

    ft_reader = FTReader("/dev/ttyUSB0", 115200)
    # ft_reader = FakeFTReader() # Require root in linux
    ft_reader_thread = threading.Thread(target=ft_reader.run_thread)
    ft_reader_thread.start()

    def signal_handler(sig, frame):
        logging.info("Ctrl+C pressed, stopping...")
        ft_reader.running = False
        ft_reader_thread.join()
        robot.home()
        logging.info("Stopped.")
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    logging.info("Warm-up ...")
    while True:

        if ft_reader.warmed_up():
            latest_data = np.asarray(ft_reader.get_filtered_latest())
            target = robot.update_pose_with_wrench(latest_data)

            # visualize the current pose, wrench, and target pose in the terminal
            print("\033[H\033[J", end="")

            print("Wrench:")
            for i in range(6):
                print(f"{i}: {make_bar(latest_data[i], 50, 1e-6)}")

            print("Current pose | Calculated target delta:")
            for i in range(6):
                print(
                    f"{i}: {make_bar(robot.last_pose[i], 50, 0.1)} \t {i}: {make_bar(target[i]-robot.last_pose[i], 50, 10)}")
            time.sleep(0.01)
