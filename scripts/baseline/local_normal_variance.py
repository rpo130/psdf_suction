import numpy as np

def metric_local_normal_variance(point_cloud_map, normals, seg, s=30):

    # get each pixel's neighbors
    h, w = point_cloud_map.shape[:2]
    idx = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'), axis=-1)     # (h,w,2)
    id_inc = np.stack(np.meshgrid(np.arange(s), np.arange(s), indexing='ij'), axis=-1).reshape(-1, 2) - s//2    # (s*s, 2)
    I = idx[..., [0]] + id_inc[None, :, 0]  # (h,w,1) + (1, s*s) = (h,w,s*s)
    J = idx[..., [1]] + id_inc[None, :, 1]

    # masking out-of-bound and not object subscribe
    valid = np.all(I >= 0, axis=-1) * np.all(I < h, axis=-1) * np.all(J >= 0, axis=-1) * np.all(J < w, axis=-1)
    valid[valid] = valid[valid] * np.all(seg[I[valid], J[valid]] == 1, axis=-1)

    # assign std
    normal_std = np.zeros_like(normals[..., 0])
    if np.any(valid):
        normal_std[valid] = normals[I[valid], J[valid]].std(axis=-2).mean(-1)

    # get score
    score = np.zeros_like(normal_std)
    if np.any(valid):
        score[normal_std == 0] = 1.0
        if np.any(normal_std > 0):
            score[normal_std > 0] = 1 - normal_std[normal_std > 0]/normal_std.max()
        score[~valid] = 0
    return score

