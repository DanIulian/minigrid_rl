import numpy as np
from typing import List

import cv2

from analytics.utils import gen_spiral

MAX_GRID_CAP = 400
SEPARATOR_SIZE = 20
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.7
FONT_TH = 1
FONT_COLOR = (255, 255, 255)


class VisualizeEpisode:
    def __init__(self, cfg):
        self.grid_split = getattr(cfg, "grid_split", 25)
        self.grid_padding = getattr(cfg, "grid_padding", 1)
        self.map_view_size = getattr(cfg, "view_size", 200)
        self.path_color_range = np.array(
            getattr(cfg, "path_color_range", [[0, 0, 255], [255, 255, 255]]))
        self.path_thickness = getattr(cfg, "path_thickness", 1)
        self.actions_draw_separate = getattr(cfg, "actions_draw_separate", [0, 1, 2, 3, 4, 5])

        self.spiral_coord = np.array(list(gen_spiral(MAX_GRID_CAP)))

    def draw_single_episode(self, map_img: np.ndarray, step_coord: np.ndarray,
                            max_coord: np.ndarray, actions: np.ndarray):
        map_view_size = self.map_view_size
        actions_draw_separate = self.actions_draw_separate

        scale = map_view_size / float(max(map_img.shape[:2]))
        map_img = cv2.resize(map_img, (0, 0), fx=scale, fy=scale)

        header = np.zeros((SEPARATOR_SIZE, map_img.shape[1] + SEPARATOR_SIZE, 3), dtype=np.uint8)
        headers = []

        def add_header(h: List, text: str):
            new_header = header.copy()
            cv2.putText(new_header, text, (0, SEPARATOR_SIZE), FONT, FONT_SIZE, FONT_COLOR,
                        FONT_TH, cv2.LINE_AA)
            h.append(new_header)

        col_imgs = []

        new_coord, new_max_coord = self.transform_unique_coord(step_coord, max_coord)

        # Draw path
        path_img = self.draw_path(map_img.copy(), new_coord, new_max_coord)
        col_imgs.append(path_img)
        add_header(headers, "Path")

        # Draw action heat maps

        heatmaps = []
        heatmap_headers = []
        unq, cnt = np.unique(step_coord, axis=0, return_counts=True)
        max_cnt = cnt.max()
        visit_cnt = 255. / max_cnt
        for action in actions_draw_separate:
            select = step_coord[actions == action]

            heat = self.get_heatmap(map_img.copy(), select, max_coord, visit_cnt=visit_cnt)
            heatmaps.append(heat)
            add_header(heatmap_headers, f"{action}")

        col_imgs += heatmaps
        headers += heatmap_headers

        separator = np.zeros((path_img.shape[0], SEPARATOR_SIZE, 3), dtype=np.uint8)
        separators = [separator] * len(col_imgs)
        img_list = [val for pair in zip(col_imgs, separators) for val in pair]

        full_img = np.row_stack([np.column_stack(headers), np.column_stack(img_list)])
        return full_img

    def get_heatmap(self, map_img: np.ndarray, step_coord: np.ndarray, max_coord: np.ndarray,
                    visit_cnt: int = 1):

        heat_map = np.zeros(max_coord)
        for row, col in step_coord:
            heat_map[row, col] += visit_cnt

        heatmap_img = cv2.applyColorMap(heat_map.astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_img = cv2.resize(heatmap_img, tuple(map_img.shape[:2][::-1]))
        fin = cv2.addWeighted(heatmap_img, 0.7, map_img, 0.3, 0)
        return fin

    def draw_path(self, img: np.ndarray, coords: np.ndarray, max_coord: np.ndarray):
        path_color_range = self.path_color_range
        path_th = self.path_thickness

        new_coord = (coords * img.shape[:2] / max_coord).astype(np.int)

        nc = new_coord.shape[0]
        colors = np.array([np.linspace(x, y, nc) for x, y in path_color_range.T]).T.astype(np.uint8)

        n_img = img
        new_coord = new_coord[:, [1, 0]]
        for i in range(nc - 1):
            n_img = cv2.line(n_img, tuple(new_coord[i]), tuple(new_coord[i+1]),
                             tuple(colors[i].tolist()), thickness=path_th)
            n_img = cv2.circle(n_img, tuple(new_coord[i]), 2, tuple(colors[i].tolist()),
                               thickness=-1)

        i = nc - 2
        n_img = cv2.circle(n_img, tuple(new_coord[i+1]), 2, tuple(colors[i].tolist()),
                           thickness=-1)

        return n_img

    def transform_unique_coord(self, coord: np.ndarray, max_coord: np.ndarray):

        grid_split = self.grid_split
        grid_padding = self.grid_padding

        unq, cnt = np.unique(coord, axis=0, return_counts=True)
        max_cnt = cnt.max()
        grid_split_s = int(np.ceil(np.sqrt(max(grid_split, max_cnt))) + grid_padding)
        grid_split_s = grid_split_s+1 if grid_split_s % 2 == 1 else grid_split_s

        occupancy_grid = np.zeros(max_coord)

        # Sub-grid coords
        no_cells = grid_split_s ** 2
        max_without_padding = no_cells - (grid_split_s - np.arange(grid_padding)*2) * 4 - 4

        # # Get spiral coord
        # spiral_coord = self.spiral_coord
        # subgrid_coord = spiral_coord[:no_cells] + np.array([grid_split_s//2] * 2)

        # Get consecutive cells
        subgrid_coord = np.array(list(np.ndindex(grid_split_s-grid_padding,
                                                 grid_split_s-grid_padding))) + grid_padding

        new_size = max_coord * grid_split_s

        new_coord = np.zeros_like(coord)
        for idx in range(coord.shape[0]):
            crt = coord[idx]
            subgrid_idx = subgrid_coord[int(occupancy_grid[crt[0], crt[1]])]  # Organized cells

            # subgrid_idx = subgrid_coord[np.random.randint(max_without_padding)] # random cell

            new_coord[idx] = crt * grid_split_s + subgrid_idx
            occupancy_grid[crt[0], crt[1]] += 1

        return new_coord, new_size


if __name__ == "__main__":

    import argparse
    n_cfg = argparse.Namespace()
    viz = VisualizeEpisode(n_cfg)

    nimg = cv2.imread("data/empty-env.png")
    map_shape = np.array([8, 8])
    ncoord = np.array([[1, 1], [1, 2], [1, 2], [1, 2], [1, 3], [2, 3], [1, 3]])
    nactions = np.array([0, 1, 1, 2, 0, 1, 3])

    new_img = viz.draw_single_episode(nimg, ncoord, map_shape, nactions)

    cv2.imshow("Map", new_img)
    cv2.waitKey(0)
