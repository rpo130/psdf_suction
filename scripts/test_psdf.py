from psdf import PSDF
from configs import config
import numpy as np

def test_fuse_point():
    psdf = PSDF(config.volume_shape, config.volume_resolution)
    psdf.fuse_point(np.array([0.0,-0.25,0.02]), config.T_world_to_volume)
    print(psdf.sdf == 0)
    print(np.where(psdf.sdf.cpu().numpy() == 0))

def test_fuse_contact():
    psdf = PSDF(config.volume_shape, config.volume_resolution)
    T = np.eye(4)
    T[:3,3] = [0.2,0.2,0.2]
    print(T)
    psdf.fuse_contact(T)
    print(psdf.sdf[(psdf.sdf > -1e-1) * (psdf.sdf < 1e-1)])

if __name__ == "__main__":
    test_fuse_contact()