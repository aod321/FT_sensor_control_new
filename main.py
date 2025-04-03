import time
from collections import deque
import threading
import logging
import signal
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from dummy import DummyRobot
from ft_reader import FTReader, B_PRIME_MAT
# from fake_ft_reader import FakeFTReader


# TUI visualizer for wrench or raw data
def make_bar(data: np.ndarray, width: int = 50, scale: int = 1) -> None:
    bar_length = min(abs(int(data * scale)), width)

    if data >= 0:
        disp = f"{' ' * width}[{(data * scale):.1f}\t]{'#' * bar_length}{' ' * (width-bar_length)}"
    else:
        disp = f"{' ' * (width-bar_length)}{'#' * bar_length}[{(data * scale):.1f}\t]{' ' * (width)}"
    return disp


def print_bars(data: np.ndarray, width: int = 50, scale: int = 1) -> None:
    """ Visualize wrench """
    print("\033[H\033[J", end="")  # Clear the screen
    for i in range(len(data)):
        print(f"{i}: {make_bar(data[i], width, scale)}")
    print(f"{str(data)}")


# Real-time plot for wrench or raw data
def show_dynamic_plot(ft_reader_instance: FTReader, b_prime_mat: np.ndarray, raw: bool = False, len: int = 400) -> None:
    """ Plot wrench or raw data in real-time """

    wrench_lab = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    raw_lab = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6"]

    plot_queue = deque(maxlen=len)
    for i in range(len):
        plot_queue.append(np.zeros(6))

    fig, ax = plt.subplots()

    x = np.arange(0, len, 1)
    lines = []
    for i in range(6):
        line, = ax.plot(
            x, list(np.array(plot_queue).T[i]), label=wrench_lab[i] if not raw else raw_lab[i])
        lines.append(line)

    if raw:
        ax.set_title("Raw data")
        ax.set_ylim(-0.04, 0.04)
    else:
        ax.set_title("Force and Torque")
        ax.set_ylim(-50, 50)

    ax.legend(loc="upper right")

    def animate(i):
        latest_data = np.asarray(ft_reader_instance.get_filtered_latest())
        if raw:
            plot_queue.append(latest_data)
        else:
            wrench = np.dot(b_prime_mat, latest_data)
            plot_queue.append(wrench)

        for i in range(6):
            lines[i].set_ydata(list(np.array(plot_queue).T[i]))

        return lines

    ani = animation.FuncAnimation(
        fig, animate, interval=10, blit=True, save_count=len)

    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Uncomment the following lines to enable robot control
    # robot = DummyRobot(real_device=True)
    # time.sleep(1)
    # print(robot.read_eef_pose())

    ft_reader = FTReader("/dev/ttyACM0", 230400)
    # ft_reader = FakeFTReader() # Mock ft sensor for control debugging, require root in linux
    ft_reader_thread = threading.Thread(target=ft_reader.run_thread)
    ft_reader_thread.start()

    # Process Ctrl-C
    def signal_handler(sig, frame):
        logging.info("Ctrl+C pressed, stopping...")
        ft_reader.running = False
        ft_reader_thread.join()
        # robot.home()
        logging.info("Stopped.")
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # wait for initial values and calibrate offset
    logging.info("Warm-up ...")
    while not ft_reader.warmed_up():
        pass
    ft_reader.set_zero()
    time.sleep(1)
    logging.info("Warm-up done.")

    # Uncomment the following lines to plot real-time data
    show_dynamic_plot(ft_reader, B_PRIME_MAT, raw=True)

    # Uncomment the following lines ro dump data to file. Use analyze.py to visualize
    # logging.info("record start")
    # while len(ft_reader.data_buffer) < 200:
    #     pass
    # time.sleep(10)
    # logging.info("record ended")
    # with open("data.pkl", "wb") as f:
    #     pickle.dump(ft_reader.data_buffer, f)
    # exit(0)

    # Uncomment the following lines to enable robot control
    # while True:
    #     if ft_reader.warmed_up():
    #         latest_data = np.asarray(ft_reader.get_filtered_latest())
    #         wrench = np.dot(B_PRIME_MAT, latest_data)
    #         target = robot.update_pose_with_wrench(latest_data)

    #         # visualize the current pose, wrench, and target pose in the terminal
    #         print("\033[H\033[J", end="")

    #         print("Wrench:")
    #         for i in range(6):
    #             print(f"{i}: {make_bar(latest_data[i], 50, 1e-6)}")

    #         print("Current pose | Calculated target delta:")
    #         for i in range(6):
    #             print(
    #                 f"{i}: {make_bar(robot.last_pose[i], 50, 0.1)} \t {i}: {make_bar(target[i]-robot.last_pose[i], 50, 10)}")
    #         time.sleep(0.01)
