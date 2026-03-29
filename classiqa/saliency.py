import torch
import torch.nn as nn
import itertools
import scipy
import skimage.filters as skfilters
import numpy as np
import cv2


class CovSal:
    """Covariance Saliency method by Erdem and Erdem, 2013:
    https://web.cs.hacettepe.edu.tr/~erkut/projects/CovSal/
    """

    def __init__(
        self,
        block_sizes=[8, 16, 32, 64, 128],
        r=3,
        m=0.1,
        eps=1e-10,
        sigma_points=False,
        alpha=np.sqrt(2),
        center_bias=False,
    ):

        self.block_sizes = block_sizes
        self.unfold_regions = {
            bs: nn.Unfold(kernel_size=bs, stride=bs) for bs in self.block_sizes
        }

        self.r = r
        self.nn_ksize = 2 * r + 1
        self.unfold_nn = nn.Unfold(self.nn_ksize, stride=1, padding=self.r)

        self.m = int(m * (self.nn_ksize**2 - 1))
        self.eps = eps
        self.sigma_points = sigma_points
        self.alpha = alpha
        self.center_bias = center_bias

    def _get_sigma_points(self, cov, ftrs_block):
        n_dims = ftrs_block.shape[-1]
        l = np.linalg.cholesky(cov)
        sigma_pts = np.concat(
            (
                self.alpha * np.sqrt(n_dims) * l,
                -self.alpha * np.sqrt(n_dims) * l,
            ),
            axis=1,
        )
        mu = ftrs_block.mean((1, 2))
        psi = np.concat((mu, sigma_pts.flatten()))
        return psi

    def compute(self, x):

        # Generate the features (F):
        # 1. LAB values
        # 2. Edge orientation
        # 3. Pixel coordinates (s1, s2)
        x_lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
        f = np.array([[-1, 0, 1]], dtype=float)
        x_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        h, w = x_gray.shape
        edges_x = np.abs(cv2.filter2D(x_gray, -1, f))
        edges_y = np.abs(cv2.filter2D(x_gray, -1, f.T))
        c1, c2 = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))
        F = np.stack([*cv2.split(x_lab), edges_x, edges_y, c1, c2], -1).astype(float)
        n_dims = F.shape[-1]  # there are 7 features in total

        # We use first and second order integral images to compute the covariance matrices
        # as in "Region Covariance: A Fast Descriptor for Detection and Classification"
        # (Tuzel, Porikli and Meer, 2006)
        # https://www.porikli.com/mysite/pdfs/porikli%202006%20-%20Region%20Covariance%20A%20Fast%20Descriptor%20for%20Detection%20and%20Classification.pdf
        fo_ii = np.zeros((h + 1, w + 1, n_dims))
        for i in range(n_dims):
            fo_ii[:, :, i] = cv2.integral(F[:, :, i])
        so_ii = np.zeros((h + 1, w + 1, n_dims, n_dims))
        for i, j in itertools.product(range(n_dims), range(n_dims)):
            so_ii[:, :, i, j] = cv2.integral(F[:, :, i] * F[:, :, j])

        # Check out this repo for region covariance:
        # https://github.com/Mrrmm510/ManifoldOptimization/tree/master/modules/region_covariance
        # https://github.com/Mrrmm510/ManifoldOptimization/blob/master/notebook/region-covariance/object-detection.ipynb

        def get_sum(ii, r1, c1, r2, c2):
            """Computes the sum inside a region of an integral image
            ii: integral image
            (r1, c1): row and column indices of the top-left corner
            (r2, c2): row and column indices of the bottom-right corner
            """
            return ii[r2 + 1, c2 + 1] + ii[r1, c1] - ii[r1, c2 + 1] - ii[r2 + 1, c1]

        # Compute covariance matrices in each region
        saliency_map = np.ones((h, w))
        for bs in self.block_sizes:

            # Divide the image into blocks
            t = torch.from_numpy(F).unsqueeze(0).permute(0, 3, 1, 2)
            ftrs_regions = self.unfold_regions[bs](t).permute(0, 2, 1).squeeze()
            ftrs_regions = ftrs_regions.view(ftrs_regions.shape[0], n_dims, bs, bs)

            # Computing the center bias
            h_tiled, w_tiled = (h // bs, w // bs)
            if self.center_bias:
                # TODO: Review - I think this is not working
                region_centers = (
                    ftrs_regions.mean([2, 3])[:, -2:]
                    .reshape(h_tiled, w_tiled, 2)
                    .numpy()
                )
                img_center = np.array([w // 2, h // 2])
                region_center_dists = np.linalg.norm(
                    region_centers - img_center, axis=2
                )
                center_bias = 1 - (region_center_dists / region_center_dists.max())
                center_bias = center_bias.reshape(-1)
            else:
                center_bias = np.ones(h_tiled * w_tiled)

            covs = []
            # Calculate grid of top-left corners
            for r in range(0, h - bs + 1, bs):
                for c in range(0, w - bs + 1, bs):
                    # Region: [r, r+bs-1] x [c, c+bs-1]
                    p = get_sum(fo_ii, r, c, r + bs - 1, c + bs - 1)
                    q = get_sum(so_ii, r, c, r + bs - 1, c + bs - 1)

                    n = bs**2
                    p = p.reshape(n_dims, 1)
                    cov_j = (q - (p @ p.T) / n) / (n - 1)
                    cov_j += np.eye(n_dims) * self.eps
                    covs.append(cov_j)

            covs_grid = np.stack(covs).reshape(-1, n_dims, n_dims)
            num_blocks = covs_grid.shape[0]

            # Once we have all the covariance matrices for the current scale,
            # we will compute their distances within a neighbourhood
            sm = np.zeros((h_tiled, w_tiled))
            num_neighbours = (self.nn_ksize**2) - 1

            region_idxs = (
                torch.arange(num_blocks)
                .unsqueeze(0)
                .reshape(h_tiled, w_tiled)
                .unsqueeze(0)
                .float()
            )
            neighbourhoods = (
                self.unfold_nn(region_idxs)
                .reshape(self.nn_ksize, self.nn_ksize, -1)
                .int()
            )

            # We will populate two arrays: the one that stores the distances between
            #  each pair of blocks, and the one that stores the overall saliency of a region
            # For the last one, we will use 123 as a placeholder. Later, when we sort the distances
            # in ascending order, we will be able deal with blocks that don't have
            # all their neighbours available (the blocks that are in the corners of the image)
            dists = np.ones((num_blocks, num_blocks)) * -1
            region_dists = np.ones((h_tiled, w_tiled, num_neighbours)) * 123
            for i in range(num_blocks):
                r = i // w_tiled
                c = i % w_tiled

                # Define neighborhood boundaries
                r_start = max(r - self.r, 0)
                r_end = min(r + self.r + 1, h_tiled)
                c_start = max(c - self.r, 0)
                c_end = min(c + self.r + 1, w_tiled)

                nbhd = neighbourhoods[
                    r_start - (r - self.r) : r_end - (r - self.r),
                    c_start - (c - self.r) : c_end - (c - self.r),
                    i,
                ]

                nb_idxs = nbhd.flatten().tolist()
                nb_idxs.remove(i)
                for i_nn in nb_idxs:
                    j = nb_idxs.index(i_nn)

                    # We only run this if d(C1, C2) has not been computed before
                    # (because d(C1, C2) = d(C2, C1))
                    if dists[i, i_nn] == -1:
                        cov_query = covs_grid[i, ...]
                        cov_nn = covs_grid[i_nn, ...]

                        # Coordinates of the center of the blocks
                        r_ii = i_nn // w_tiled
                        c_ii = i_nn % w_tiled
                        x_i = (bs * (c + 1)) - (bs // 2)
                        y_i = (bs * (r + 1)) - (bs // 2)
                        x_i_nn = (bs * (c_ii + 1)) - (bs // 2)
                        y_i_nn = (bs * (r_ii + 1)) - (bs // 2)
                        center_dist = np.linalg.norm((x_i_nn - x_i, y_i_nn - y_i))

                        # Compute distance
                        if not self.sigma_points:
                            evs = scipy.linalg.eigvals(cov_query, cov_nn)
                            d = np.sqrt(np.sum(np.log(np.real(evs) + self.eps) ** 2))
                        else:
                            # Encoding the query and the neighbour
                            psi_query = self._get_sigma_points(
                                cov_query, ftrs_regions[i]
                            )
                            psi_nn = self._get_sigma_points(cov_nn, ftrs_regions[i_nn])
                            d = np.linalg.norm(psi_query - psi_nn)
                        d /= 1 + center_dist
                        d *= center_bias[i]

                        dists[i, i_nn] = d
                        dists[i_nn, i] = d

                    region_dists[r, c, j] = dists[i, i_nn]

            # Some blocks don't have all neighbours available (e.g. in the image corners)
            region_dists = np.sort(region_dists, axis=-1)[..., : self.m]
            valid_mask = (region_dists != 123).astype(float)
            sm = (region_dists * valid_mask).sum(-1) / valid_mask.sum(-1)
            # sm /= sm.max()

            sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
            saliency_map *= sm

        saliency_map = skfilters.gaussian(saliency_map, sigma=int(0.02 * w))

        return saliency_map


if __name__ == "__main__":

    img = cv2.imread("classiqa/tmp/saliency/fish.png")
    ratio = 512 / max(img.shape[:2])
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    covsal = CovSal(sigma_points=True, center_bias=True)
    covsal.compute(img)
