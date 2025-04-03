import fibre
import logging
import numpy as np
import scipy


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
                self.arm = fibre.find_any(serial_number=serial_id)
            else:
                self.arm = fibre.find_any()
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
