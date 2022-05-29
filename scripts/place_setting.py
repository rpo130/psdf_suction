import os
import yaml


import rosparam

from scripts.config import config
from scripts.ur5_commander import UR5Commander

if __name__ == '__main__':
    arm = UR5Commander()
    output_file = "../config/positions.yaml"

    arm.set_pose(config.init_pose, True)
    rosparam.set_param("/positions_setting/init_positions", str(list(arm.get_positions())))

    print("move arm to pre place pose, and press key")
    input()
    rosparam.set_param("/positions_setting/place_pre_positions", str(list(arm.get_positions())))

    print("move arm to place pose, and press key")
    input()
    rosparam.set_param("/positions_setting/place_positions", str(list(arm.get_positions())))

    rosparam.dump_params(output_file, param="positions_setting")