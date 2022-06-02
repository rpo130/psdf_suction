import numpy as np

from configs import config

class BackgroundExtractor:
    def __init__(self) -> None:
        self.background = np.load("config/background.npz")
    
    def extract(self, point_map):
        return np.linalg.norm(point_map - self.background) >= 0.02

def build_backgound():
    I, J = np.meshgrid(
        np.arange(config.psdf_length), 
        np.arange(config.psdf_width)
    )
    points = (np.stack([I, J], axis=-1) + 0.5) * config.psdf_resolution
    plane_height = config.z_min
    points = np.concatenate([points, np.ones_like(points[..., [0]]) * plane_height])
    np.save("config/background.npz", points)

if __name__=="__main__":
    build_backgound