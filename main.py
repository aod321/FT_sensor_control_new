import serial
import time
from collections import deque
import threading
import logging
import signal
import numpy as np
import scipy


B_PRIME_MAT = np.array([
    [-8, 62, -59, 4, -912, 914],
    [48, -25, -35, -1058, 528, 529],
    [600, 600, 600, 0, 0, 0],
    [-5, 478, -468, -128, 67, -56],
    [550, -268, -278, -5, -107, 116],
    [14, 16, 19, -288, -288, -289]
])

def make_bar(data: np.ndarray, width: int = 50, scale:int=1e-6) -> None:
    bar_length = min(abs(int(data * scale)), width)

    if data >= 0:
        disp = f"{' ' * width}[{data:.1f}\t]{'#' * bar_length}{' ' * (width-bar_length)}"
    else:
        disp = f"{' ' * (width-bar_length)}{'#' * bar_length}[{data}\t]{' ' * (width)}"
    return disp
    

def print_bars(data: np.ndarray, width: int = 50, scale:int=1e-6) -> None:
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
    def __init__(self, real_device=False, serial_port="/dev/ttyACM0"):
        self.real_device = real_device
        if real_device:
            self.ser = serial.Serial(serial_port, 9600, timeout=0.1)

        logging.info(self._send_cmd_and_wait("!START\n"))
        logging.info(self._send_cmd_and_wait("!HOME\n", timeout=100))

        self.xyz_control_gain = 1e-9
        self.rpy_control_gain = 1e-9

        self.last_pose = None
        self.last_target_pose = None

        self.ignored_traget_count = 0

    def home(self):
        self._send_cmd("!HOME\n")

    def _send_cmd_and_wait(self, cmd: str, timeout:float=0.1) -> str:
        # clear the buffer
        while self.ser.inWaiting() > 0:
            self.ser.readline()

        self.ser.write(cmd.encode())
        logging.debug(f"Sent command: {cmd} , waiting for response")

        response = None
        t = time.time()
        while not response and time.time() - t < timeout:
            response = self.ser.readline().decode()
        if not response:
            logging.error("Timeout waiting for response")
            return 
        logging.debug(f"Received response: {response}")
        return response
    

    def _send_cmd(self, cmd: str) -> None:
        self.ser.write(cmd.encode())
        logging.debug(f"Sent command: {cmd}")

    def read_eef_pose(self) -> np.ndarray[6]:
        """ EEF pose is a 6D vector [x, y, z, roll, pitch, yaw] """
        if self.real_device:
            # send #GETLPOSE over serial, and wait for response
            # the serial should return "ok %.2f %.2f %.2f %.2f %.2f %.2f", corresponding to x, y, z, roll, pitch, yaw
            # return example: "ok 227.52 -0.19 323.88 0.13 90.00 0.00"
            response = self._send_cmd_and_wait("#GETLPOSE\n", timeout=1)
            if not response:
                return self.last_pose
            if response.startswith("ok"):
                try:
                    eef_pose = np.array([float(x) for x in response.split()[1:]])
                    self.last_pose = eef_pose
                    return eef_pose
                except ValueError:
                    logging.error("Failed to parse response: ", response)
                    return np.array([1, 1, 1, 0, 0, 1])
        else:
            return np.array([1, 1, 1, 0, 0, 1])

    def set_target_pose(self, target_pose: np.ndarray[6]) -> None:
        """ Set the target pose of the robot """
        # prevent setting a target that is too far from last target or current pose
        if self.last_target_pose is not None and self.last_pose is not None and self.ignored_traget_count < 5:
            if np.any(np.abs(target_pose - self.last_target_pose) > 5) or np.any(np.abs(target_pose - self.last_pose) > 5):
                logging.error("Target pose too far from last target, ignoring("+str(target_pose)+")")
                self.ignored_traget_count += 1
                return
        
        self.last_target_pose = target_pose
        self.ignored_traget_count = 0
        logging.info(f"Setting target pose to {str(target_pose[:3])}, {target_pose[3:]}")
        if self.real_device:
            result = self._send_cmd_and_wait("@"+",".join([f"{x:.2f}" for x in target_pose])+ "\n", timeout=10)
            logging.info(f"Set target pose result: {result}")

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
        R = scipy.spatial.transform.Rotation.from_euler("xyz", euler_angles, degrees=True).as_matrix()

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
        target_R = R @ scipy.linalg.expm(skew_symmetric(self.rpy_control_gain * torque_base))
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

        self.data_buffer = deque(maxlen=50)  # Buffer for storing the last 200 data points
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

    def run_thread(self):
        SG_ORDER = [1, 6, 5, 4, 3, 2]
        interval_sum = 0.0  # Sum of intervals measured
        interval_count = 0  # Count of intervals measured
        last_print_time = time.time()  # Last time the average was printed

        self.running = True
        while self.running:
            if self.serial_port.inWaiting() > 0:
                current_time = time.time()
                actual_interval = (current_time - self.last_update_time) * 1000  # ms
                self.last_update_time = current_time

                # Accumulate the sum and count of intervals
                interval_sum += actual_interval
                interval_count += 1

                try:
                    line = self.serial_port.readline().decode().strip()
                    if line:  # If the line is not empty
                        try:
                            # Attempt to parse the line as a list of floats
                            sensor_values = [float(val) for val in line.split(",")[:6]]

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

                            self.data_buffer.append(np.dot(B_PRIME_MAT, reordered_values))
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
                        logging.debug("Average serial interval: %.2f ms", average_interval)
                    else:
                        logging.debug("No data received in the last second.")

                    # Reset for the next second
                    last_print_time = current_time
                    interval_sum = 0.0
                    interval_count = 0



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    robot = DummyRobot(real_device=True)
    time.sleep(1)
    print(robot.read_eef_pose())

    ft_reader = FTReader("/dev/ttyUSB0", 115200)
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
                print(f"{i}: {make_bar(robot.last_pose[i], 50, 0.1)} \t {i}: {make_bar(target[i]-robot.last_pose[i], 50, 10)}")
            
            time.sleep(0.1)