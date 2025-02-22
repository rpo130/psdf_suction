import pyrealsense2 as rs
import numpy as np

class RealSenseCommander(object):
    def __init__(self):
        self.cam_sr = "043322070286"
        self.pipe = rs.pipeline(rs.context())
        self.config = rs.config()
        self.config.enable_device(self.cam_sr)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)

    def start(self):
        self.pipe.start(self.config)

    def get_image(self):
        frames = self.pipe.wait_for_frames()
        algned_frames = self.align.process(frames)
        color = np.array(algned_frames.get_color_frame().get_data(), dtype=np.uint8)
        depth = np.array(algned_frames.get_depth_frame().get_data(), dtype=np.float32) / 1000
        depth[depth > 1] = 0
        return color, depth