import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import pyrealsense2 as rs

class Recorder(object):
    def __init__(self, output_dir=None):
        if output_dir is None:
            self.output_dir = os.path.dirname(__file__)
        else:
            os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir
        self.output_file = os.path.join(self.output_dir, "result_and_time.txt")
        self.log_file = os.path.join(self.output_dir, "log.txt")
        self.recoder_cam_sr = "619206001621"
        self.pipe = rs.pipeline(rs.context())
        self.config = rs.config()
        self.config.enable_device(self.recoder_cam_sr)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self.cnt = 0

    def start(self):
        self.pipe.start(self.config)

    def get_image(self):
        frames = self.pipe.wait_for_frames()
        color = np.array(frames.get_color_frame().get_data())
        return color

    def set_target_region(self):
        img = self.get_image()
        fig = plt.figure("recorder_test")
        plt.imshow(img)
        self._ax = plt.axes()
        self._canvas = fig.canvas
        self.base = [0, 0]
        self.offset = [img.shape[0]-1, img.shape[1]-1]
        self._pressing = False
        self._lines = [plt.Line2D([self.base[1], self.offset[1]],[self.base[0], self.offset[0]], color=[1, 0, 0, 1]) for _ in range(4)]
        def on_click(event):
            if event.xdata != None and event.ydata != None:
                # print(event.xdata, event.ydata)
                self._pressing = True
                self.base = [int(event.ydata), int(event.xdata)]
        def on_release(event):
            if event.xdata != None and event.ydata != None:
                # print(event.xdata, event.ydata)
                self._pressing = False
                self.offset = [int(event.ydata)-self.base[0], int(event.xdata)-self.base[1]]
        def mouse_move(event):
            img = self.get_image()
            self._ax.imshow(img)
            if self._pressing:
                self._lines[0].set_xdata([self.base[1], event.xdata])
                self._lines[0].set_ydata([self.base[0], self.base[0]])
                self._lines[1].set_xdata([event.xdata, event.xdata])
                self._lines[1].set_ydata([self.base[0], event.ydata])
                self._lines[2].set_xdata([self.base[1], event.xdata])
                self._lines[2].set_ydata([event.ydata, event.ydata])
                self._lines[3].set_xdata([self.base[1], self.base[1]])
                self._lines[3].set_ydata([self.base[0], event.ydata])
                for line in self._lines:
                    self._ax.add_line(line)
            self._canvas.draw()
        self._canvas.mpl_connect("button_press_event", on_click)
        self._canvas.mpl_connect("button_release_event", on_release)
        self._canvas.mpl_connect("motion_notify_event", mouse_move)
        plt.show()
        # self.pattern = cv2.cvtColor(cv2.blur(img[self.base[0]:self.base[0]+self.offset[0], self.base[1]:self.base[1]+self.offset[1]], (5, 5)), cv2.COLOR_RGB2GRAY)
        for i in range(50):
            self.get_image()
        img = self.get_image()
        self.pattern = img[self.base[0]:self.base[0]+self.offset[0], self.base[1]:self.base[1]+self.offset[1]]
        cv2.imwrite(os.path.join(self.output_dir, "pattern.png"), self.pattern)

    def set(self):
        self.start_time = time.time()

    def check(self, threshold=0.1):
        cost_time = time.time() - self.start_time
        for i in range(50):
            self.get_image()
        img = self.get_image()
        target = img[self.base[0]:self.base[0]+self.offset[0], self.base[1]:self.base[1]+self.offset[1]]
        cv2.imwrite(os.path.join(self.output_dir, "target_{}.png".format(self.cnt)), target)
        diff = (target.astype(np.float) - self.pattern.astype(np.float)) / 255
        score = np.linalg.norm(diff) / min(self.offset[0], self.offset[1])
        with open(self.output_file, 'a') as f:
            print(int(score > threshold), cost_time, file=f)
        self.cnt += 1
        return score > threshold, score

    def log(self, *args):
        with open(self.log_file, 'a') as f:
            print(*args, file=f)
            print(*args)