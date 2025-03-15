import gc
import threading
import time
import numpy as np

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.configs import ManipulatorRobotConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
# from manipulator import ManipulatorRobot
import torch
from pymycobot import MyCobot320Socket


class MycobotManipulator:
    def __init__(self, ip, port, config: ManipulatorRobotConfig):
        self.config = config
        self.mc = None
        self.ip = ip
        self.port = port
        self.robot_type = self.config.type
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.logs = {}

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        names = ["1", "2", "3", "4", "5", "6", "7"]
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(names),),
                "names": names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(names),),
                "names": names,
            },
        }

    def connect(self):
        self.mc = MyCobot320Socket(self.ip, self.port)
        self.mc.send_angles([0, 0, 0, 0, 0, 0], 40)
        ret_mode = self.mc.set_gripper_mode(0)
        time.sleep(1)
        self.mc.set_gripper_state(0, 100)
        self.mc.set_gripper_state(254, 100)

        self.gripper_open_close_threshold = None
        while not self.gripper_open_close_threshold:
            self.gripper_open_close_threshold = self.mc.get_gripper_value() - 10
        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()
        self.is_connected = True
        print("connect to your mycobot.")
        return


    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        # TODO
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )
        action_sent = []
        goal_pos = action
        action_sent.append(goal_pos)
        goal_pos = goal_pos.numpy().astype(np.float32).tolist()
        print(goal_pos)
        self.mc.send_angles(goal_pos[:-1], 40)
        self.mc.set_gripper_state(1 if goal_pos[-1] >= 0.5 else 0, 40)
        return torch.cat(action_sent)


    def disconnect(self):
        # TODO
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )
        self.mc = None
        gc.collect()
        self.is_connected = False
