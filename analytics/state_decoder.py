"""
    part of code from https://github.com/maximecb/gym-minigrid
    @ modifications so as to render without environment @
"""

import gym
import gym_minigrid
from gym_minigrid.rendering import Renderer
from gym_minigrid.minigrid import CELL_PIXELS
from gym_minigrid.minigrid import Grid
import numpy as np


def get_array_from_pixmap(pix_map):
    """
    Get a numpy array of RGB pixel values.
    The size argument should be (3,w,h)
    """

    width, height = pix_map.width, pix_map.height
    shape = (height, width, 3)

    num_bytes = width * height * 3
    buf = pix_map.img.bits().asstring(num_bytes)
    output = np.frombuffer(buf, dtype='uint8')
    output = output.reshape(shape)

    return output


class StateDecoder:
    def __init__(self, state_size, agent_view_size=7):
        self.width, self.height = state_size
        self.agent_view_size = agent_view_size
        self.obs_render = None
        self.grid_render = None
        self._render = False

    def get_obs_render(self, obs, tile_pixels=CELL_PIXELS // 2, mode='rgb_array'):
        """
        Render an agent observation for visualization
        """

        if self.obs_render is None:
            obs_render = Renderer(
                self.agent_view_size * tile_pixels,
                self.agent_view_size * tile_pixels,
                self._render
            )
            self.obs_render = obs_render
        else:
            obs_render = self.obs_render

        r = obs_render

        r.beginFrame()

        grid = Grid.decode(obs)

        # Render the whole grid
        grid.render(r, tile_pixels)

        # Draw the agent
        ratio = tile_pixels / CELL_PIXELS
        r.push()
        r.scale(ratio, ratio)
        r.translate(
            CELL_PIXELS * (0.5 + self.agent_view_size // 2),
            CELL_PIXELS * (self.agent_view_size - 0.5)
        )
        r.rotate(3 * 90)
        r.setLineColor(255, 0, 0)
        r.setColor(255, 0, 0)
        r.drawPolygon([
            (-12, 10),
            (12, 0),
            (-12, -10)
        ])
        r.pop()

        r.endFrame()

        if mode == 'rgb_array':
            return get_array_from_pixmap(r)
        elif mode == 'pixmap':
            return r.getPixmap()

        return r.getPixmap()

    def get_state_render(self, state, agent_locations, tile_pixels=CELL_PIXELS // 2,
                         mode='rgb_array',):
        crt_state = state.copy()
        agent_pos = tuple(agent_locations)
        agent_dir = crt_state[agent_pos][1]
        carying = crt_state[agent_pos][2]
        crt_state[agent_pos][0] = carying
        crt_state[agent_pos][1] = 0
        return self.render(agent_pos, agent_dir, crt_state, tile_pixels=tile_pixels, mode=mode)

    def render(self, agent_pos, agent_dir, state, mode='rgb_array',
               tile_pixels=CELL_PIXELS // 2, close=False):
        """
        Render the whole-grid human view
        """

        if self.grid_render is None:
            grid_render = Renderer(
                self.width * CELL_PIXELS,
                self.height * CELL_PIXELS,
                self._render
            )
            self.grid_render = grid_render
        else:
            grid_render = self.grid_render

        r = grid_render

        r.beginFrame()

        grid = Grid.decode(state)

        # Render the whole grid
        grid.render(r, CELL_PIXELS)

        # # Render the whole grid
        # self.grid.render(r, CELL_PIXELS)

        # Draw the agent
        r.push()
        r.translate(
            CELL_PIXELS * (agent_pos[0] + 0.5),
            CELL_PIXELS * (agent_pos[1] + 0.5)
        )
        r.rotate(agent_dir * 90)
        r.setLineColor(255, 0, 0)
        r.setColor(255, 0, 0)
        r.drawPolygon([
            (-12, 10),
            (12, 0),
            (-12, -10)
        ])
        r.pop()

        # # Compute which cells are visible to the agent
        # _, vis_mask = self.gen_obs_grid()

        # Compute the absolute coordinates of the bottom-left corner
        # of the agent's view area
        # f_vec = self.dir_vec
        # r_vec = self.right_vec
        # top_left = self.agent_pos + f_vec * (self.agent_view_size - 1) - r_vec * (
        #             self.agent_view_size // 2)
        #
        # # For each cell in the visibility mask
        # for vis_j in range(0, self.agent_view_size):
        #     for vis_i in range(0, self.agent_view_size):
        #         # If this cell is not visible, don't highlight it
        #         if not vis_mask[vis_i, vis_j]:
        #             continue
        #
        #         # Compute the world coordinates of this cell
        #         abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
        #
        #         # Highlight the cell
        #         r.fillRect(
        #             abs_i * CELL_PIXELS,
        #             abs_j * CELL_PIXELS,
        #             CELL_PIXELS,
        #             CELL_PIXELS,
        #             255, 255, 255, 75
        #         )

        r.endFrame()

        if mode == 'rgb_array':
            return get_array_from_pixmap(r)
        elif mode == 'pixmap':
            return r.getPixmap()

        return r

