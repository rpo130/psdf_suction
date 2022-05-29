import pcl
import numpy as np
import time
import torch


# device = "cpu"
def metric_planar_distance(points, r=0.01, threshold=3e-6, tolerance=0.0025, min_size=500, device = "cuda:0"):

    # PCL clutter segmentation
    ts = time.time()
    cloud = pcl.PointCloud(points.astype(np.float32))
    tree = cloud.make_kdtree()
    ec = cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(tolerance)
    ec.set_MinClusterSize(min_size)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    print(len(cluster_indices))
    # print("segmentation time", time.time()-ts)

    # analyze each point
    grasps = []
    distances = []
    ts = time.time()
    for idx in cluster_indices:
        obj = torch.from_numpy(points[idx]).to(device)

        # get local plane
        distance = torch.norm(obj[:, None, :] - obj[None, ...], dim=-1)
        neigbor_mask = distance <= 2*r
        neigbor = torch.stack([obj]*obj.shape[0], dim=0) * neigbor_mask[..., None]
        n = neigbor_mask.sum(dim=-1, keepdims=True)
        x = neigbor[..., 0].sum(dim=-1, keepdims=True)
        y = neigbor[..., 1].sum(dim=-1, keepdims=True)
        z = neigbor[..., 2].sum(dim=-1, keepdims=True)
        xy = (neigbor[..., 0] * neigbor[..., 1]).sum(dim=-1, keepdims=True)
        yz = (neigbor[..., 1] * neigbor[..., 2]).sum(dim=-1, keepdims=True)
        xz = (neigbor[..., 0] * neigbor[..., 2]).sum(dim=-1, keepdims=True)
        xx = (neigbor[..., 0] * neigbor[..., 0]).sum(dim=-1, keepdims=True)
        yy = (neigbor[..., 1] * neigbor[..., 1]).sum(dim=-1, keepdims=True)
        A = torch.cat([xx, xy, x, xy, yy, y, x, y, n], dim=-1).reshape(-1, 3, 3)
        b = torch.cat([xz, yz, z], dim=-1)
        G = torch.transpose(A, 1, 2) @ A + torch.eye(3).float().to(device) * 1e-10
        h = torch.transpose(A, 1, 2) @ b[..., None]
        v = torch.solve(h, G).solution[..., -1]

        # compute MSE
        base = (v[..., 0]**2 + v[..., 1]**2 + 1)**0.5
        err = (- v[..., [0]] * neigbor[..., 0] - v[..., [1]] * neigbor[..., 1] + neigbor[..., 2] - v[..., [2]]) / base[..., None]
        err[~neigbor_mask] = 0
        err = (err**2).sum(dim=-1) / n[..., -1]
        sorted_err, sorted_id = torch.sort(err)

        # get distance to object centroid
        index_mask = torch.arange(obj.shape[0]) <= torch.ceil(torch.FloatTensor([obj.shape[0] * 0.05]))
        choice = torch.logical_or(sorted_err < threshold, index_mask.to(device))
        # shift = torch.norm(neigbor.sum(dim=1)/n-obj, dim=-1)
        # print(shift.shape, shift.min(), shift.max())
        # choice = torch.logical_and(choice, shift < 0.005)
        sorted_obj = obj[sorted_id]
        grasp = sorted_obj[choice]
        c_obj = obj.mean(dim=0, keepdims=True)
        d = torch.norm(grasp - c_obj, dim=-1)
        grasps += grasp.cpu().numpy().tolist()
        distances += d.cpu().numpy().tolist()
    # print("analysis time", time.time() - ts)

    ts = time.time()
    id_sorted = np.argsort(distances)
    grasps = np.array(grasps)[id_sorted]
    distances = np.array(distances)[id_sorted]
    # print("ranking time", time.time()-ts)
    return grasps, distances