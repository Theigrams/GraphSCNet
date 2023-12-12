import torch
from torch import Tensor
from vision3d.ops import pairwise_distance


def prob_spacial_consistency(
    src_corr_points: Tensor,
    tgt_corr_points: Tensor,
    inlier_threshold: float,
    inlier_ratio: float = 0.1,
):
    """Compute spatial consistency by probability model.

    PSC(d) = p(inlier | d)
           = p(d | inlier) * p(inlier) / (p(d | inlier) * p(inlier) + p(d | outlier) * p(outlier))

    Args:
        src_corr_points (Tensor): The correspondence points in the source point cloud in the shape of (*, N, 3).
        tgt_corr_points (Tensor): The correspondence points in the source point cloud in the shape of (*, N, 3).
        inlier_threshold (float): The inlier threshold.
        inlier_ratio (float): The inlier ratio.
    """
    src_dist_mat = pairwise_distance(src_corr_points, src_corr_points, squared=False)
    tgt_dist_mat = pairwise_distance(tgt_corr_points, tgt_corr_points, squared=False)  # (*, N, N)
    cross_dist = torch.abs(src_dist_mat - tgt_dist_mat)
    space_size = torch.amax(cross_dist, dim=(1, 2))[:, None, None]
    posterior_inlier = (2 - cross_dist / inlier_threshold).pow(3) / (4 * inlier_threshold)
    posterior_inlier = torch.relu(posterior_inlier)
    posterior_outlier = 4 * (1 - cross_dist / space_size).pow(3) / space_size
    posterior_outlier = torch.relu(posterior_outlier)
    prob_inlier = posterior_inlier * inlier_ratio
    prob_outlier = posterior_outlier * (1 - inlier_ratio)
    psc_mat = prob_inlier / (prob_inlier + prob_outlier + 1e-8)
    return psc_mat
