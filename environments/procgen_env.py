import numpy as np
import struct
from gymnasium import spaces
from procgen import ProcgenGym3Env
from PIL import Image


# procgen discrete(15) -> our discrete(4): 0=up, 1=down, 2=left, 3=right
# procgen action layout (set_action_xy): vx = act/3 - 1, vy = act%3 - 1
#   1=LEFT(vx=-1), 3=DOWN(vy=-1), 5=UP(vy=+1), 7=RIGHT(vx=+1)
# procgen world: x=right, y=up  (rendering flips y)
# numpy grid[y, x]: y=0 is bottom of world (top of screen)
ACT_MAP = {0: 5, 1: 3, 2: 1, 3: 7}
N_ACT = 4
# action 0=up(vy+1), 1=down(vy-1), 2=left(vx-1), 3=right(vx+1)
DX = [0, 0, -1, 1]
DY = [1, -1, 0, 0]
WALL_OBJ = 51
GOAL_OBJ = 2


class _Buf:
    """Sequential reader for procgen binary state."""
    def __init__(self, data):
        self.d = data
        self.o = 0

    def i(self):
        v = struct.unpack_from('<i', self.d, self.o)[0]
        self.o += 4
        return v

    def f(self):
        v = struct.unpack_from('<f', self.d, self.o)[0]
        self.o += 4
        return v

    def s(self):
        n = self.i()
        v = self.d[self.o:self.o + n]
        self.o += n
        return v

    def skip(self, n):
        self.o += n

    def rng(self):
        self.i()  # is_seeded
        self.s()  # mt19937 state string

    def entity(self):
        x = self.f(); y = self.f()  # x, y
        self.skip(4 * 4)              # vx, vy, rx, ry
        tp = self.i()                 # type
        self.skip(4 * 2)              # image_type, image_theme
        self.i()                      # render_z
        self.skip(4 * 2)              # will_erase, collides_with_entities
        self.skip(4 * 3)              # collision_margin, rotation, vrot
        self.skip(4 * 6)              # is_reflected, fire_time, spawn_time, life_time, expire_time, use_abs_coords
        self.skip(4 * 4)              # friction, smart_step, avoids_collisions, auto_erase
        self.skip(4 * 6)              # alpha, health, theta, grow_rate, alpha_decay, climber_spawn_x
        return x, y, tp


def _extract_grid(env, idx):
    """Parse procgen maze binary state to get grid, agent pos, goal pos."""
    b = _Buf(env.callmethod("get_state")[idx])

    # --- Game::serialize ---
    b.i()              # SERIALIZE_VERSION
    b.s()              # game_name
    for _ in range(9):
        b.i()          # paint_vel_info, use_generated_assets, use_monochrome_assets,
                       # restrict_themes, use_backgrounds, center_agent,
                       # debug_mode, distribution_mode, use_sequential_levels
    for _ in range(3):
        b.i()          # use_easy_jump, plain_assets, physics_mode
    b.i()              # grid_step
    for _ in range(4):
        b.i()          # level_seed_low, level_seed_high, game_type, game_n
    b.rng()            # level_seed_rand_gen
    b.rng()            # rand_gen
    b.f()              # step_data.reward
    b.i(); b.i()       # done, level_complete
    b.i(); b.i()       # action, timeout
    for _ in range(4):
        b.i()          # current_level_seed, prev_level_seed, episodes_remaining, episode_done
    b.i()              # last_reward_timer
    b.f()              # last_reward
    b.i()              # default_action
    b.i()              # fixed_asset_seed
    b.i()              # cur_time
    b.i()              # is_waiting_for_step

    # --- BasicAbstractGame::serialize ---
    b.i()              # grid_size
    n_ents = b.i()     # entities count
    agent_x, agent_y = 0, 0
    for _ in range(n_ents):
        ex, ey, tp = b.entity()
        if tp == 0:    # PLAYER (object-ids.h: PLAYER=0)
            agent_x, agent_y = int(ex), int(ey)

    b.i()              # use_procgen_background
    b.i()              # background_index
    b.f(); b.f()       # bg_tile_ratio, bg_pct_x
    b.f()              # char_dim
    b.i(); b.i(); b.i()  # last_move_action, move_action, special_action
    b.f(); b.f(); b.f()  # mixrate, maxspeed, max_jump
    b.f(); b.f(); b.f()  # action_vx, action_vy, action_vrot
    b.f(); b.f()       # center_x, center_y
    b.i(); b.i()       # random_agent_start, has_useful_vel_info
    b.i()              # step_rand_int
    b.rng()            # asset_rand_gen
    mw = b.i()         # main_width
    mh = b.i()         # main_height
    b.i()              # out_of_bounds_object
    for _ in range(6):
        b.f()          # unit, view_dim, x_off, y_off, visibility, min_visibility

    # --- grid ---
    gw = b.i()
    gh = b.i()
    gn = b.i()         # vector length
    grid = np.array([b.i() for _ in range(gn)], dtype=np.int32).reshape(gh, gw)

    # --- MazeGame::serialize ---
    maze_dim = b.i()
    world_dim = b.i()

    nav = (grid != WALL_OBJ).astype(np.int32)
    gy, gx = np.where(grid == GOAL_OBJ)
    goal = (int(gx[0]), int(gy[0])) if len(gx) > 0 else None

    return nav, (agent_x, agent_y), goal, world_dim


def _q_iteration(nav, goal, world_dim, discount=0.99, num_itrs=200):
    """Q-value iteration on the grid. Returns optimal action map.

    Following D4RL pointmaze q_iteration.py:
      Q(s,a) = R(s,a) + discount * sum_s' T(s,a,s') * V(s')
      V(s) = max_a Q(s,a)
      opt(s) = argmax_a Q(s,a)

    States are (x, y) cells where nav[y, x] == 1.
    Actions: 0=up(dy=+1), 1=down(dy=-1), 2=left(dx=-1), 3=right(dx=+1).
    Reward: +10 at goal cell, 0 elsewhere.
    Transitions: deterministic; hitting wall stays in place.
    """
    # enumerate free cells as state indices
    free = []
    idx_map = {}
    for y in range(world_dim):
        for x in range(world_dim):
            if nav[y, x]:
                idx_map[(x, y)] = len(free)
                free.append((x, y))
    ns = len(free)
    if ns == 0:
        return np.full((world_dim, world_dim), -1, np.int32)

    gx, gy = goal
    goal_idx = idx_map.get((gx, gy), -1)

    # build transition matrix T[s, a] -> s' and reward R[s, a]
    # deterministic so T is just a next-state lookup
    next_state = np.zeros((ns, N_ACT), dtype=np.int32)
    reward = np.zeros((ns, N_ACT), dtype=np.float64)

    for si, (x, y) in enumerate(free):
        for a in range(N_ACT):
            nx, ny = x + DX[a], y + DY[a]
            if (nx, ny) in idx_map:
                next_state[si, a] = idx_map[(nx, ny)]
            else:
                next_state[si, a] = si  # wall: stay in place
            # reward for landing on goal
            nsi = next_state[si, a]
            if nsi == goal_idx:
                reward[si, a] = 10.0

    # Q-value iteration
    q = np.zeros((ns, N_ACT), dtype=np.float64)
    for _ in range(num_itrs):
        v = np.max(q, axis=1)
        q = reward + discount * v[next_state]

    # extract optimal action per cell
    opt = np.full((world_dim, world_dim), -1, dtype=np.int32)
    for si, (x, y) in enumerate(free):
        opt[y, x] = int(np.argmax(q[si]))
    return opt


def _render_opt_grid(nav, opt, agent_pos, goal_pos,
                     world_dim, img_size=512):
    """Render optimal-action arrows on the maze grid.

    Walls = black, free cells = white, arrows show optimal action,
    agent = red circle, goal = green square.
    y=0 in grid is bottom of world, so we flip y for the image.
    """
    cell = img_size // world_dim
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # arrow offsets for drawing: (dx_pixel, dy_pixel) per action
    # action 0=up(world +y = screen -row), 1=down, 2=left, 3=right
    arrow_color = (50, 120, 220)
    goal_color = (0, 200, 0)
    agent_color = (220, 40, 40)

    for gy in range(world_dim):
        for gx in range(world_dim):
            # flip y: world y=0 -> image row = (wd-1)*cell
            iy = (world_dim - 1 - gy) * cell
            ix = gx * cell
            if not nav[gy, gx]:
                # wall -> black (already 0)
                continue
            # free cell -> white background
            img[iy:iy + cell, ix:ix + cell] = 255

            a = opt[gy, gx]
            if a < 0:
                continue
            # draw arrow in cell center
            cy = iy + cell // 2
            cx = ix + cell // 2
            hl = cell // 3  # half-length
            # pixel directions (image coords)
            # up in world = -row in image
            pdx = [0, 0, -1, 1]
            pdy = [-1, 1, 0, 0]
            ex = cx + pdx[a] * hl
            ey = cy + pdy[a] * hl
            # draw line
            _draw_line(img, cx, cy, ex, ey, arrow_color, 2)
            # draw arrowhead
            _draw_circle(img, ex, ey, 3, arrow_color)

    # draw goal
    if goal_pos is not None:
        gx, gy = goal_pos
        iy = (world_dim - 1 - gy) * cell
        ix = gx * cell
        m = cell // 5
        img[iy + m:iy + cell - m, ix + m:ix + cell - m] = goal_color

    # draw agent
    if agent_pos is not None:
        ax, ay = agent_pos
        iy = (world_dim - 1 - ay) * cell + cell // 2
        ix = ax * cell + cell // 2
        _draw_circle(img, ix, iy, cell // 3, agent_color)

    return img


def _draw_line(img, x0, y0, x1, y1, color, thickness=1):
    """Bresenham line draw on numpy image."""
    h, w = img.shape[:2]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        for t in range(-thickness, thickness + 1):
            for s in range(-thickness, thickness + 1):
                py, px = y0 + t, x0 + s
                if 0 <= py < h and 0 <= px < w:
                    img[py, px] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _draw_circle(img, cx, cy, r, color):
    """Draw filled circle on numpy image."""
    h, w = img.shape[:2]
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                py, px = cy + dy, cx + dx
                if 0 <= py < h and 0 <= px < w:
                    img[py, px] = color


class VecProcgenMaze:
    """Vectorized gymnasium-style wrapper. Each sub-env shares one ProcgenGym3Env.

    Parameters
    ----------
    env : ProcgenGym3Env
        The underlying gym3 vectorized environment.
    render_mode : str or None
        If "rgb_array", full 512×512 renders are stored in infos.
    visibility : int or None
        Partial-observability window size (e.g. 7).  Observations are always
        ``(world_dim, world_dim, 3)`` float32 grids:

        * channel 0 – wall:  1 = wall, 0 = free, −1 = outside visibility
        * channel 1 – goal:  1 = goal, 0 = not goal, −1 = outside visibility
        * channel 2 – agent: 1 = agent, 0 = empty, −1 = outside visibility

        When *visibility* is ``None`` every cell is visible (full obs).
        Otherwise only a *visibility × visibility* window centred on the
        agent is revealed; all other cells are −1 in every channel.
    """

    def __init__(self, env, render_mode=None, visibility=None,
                 fixed_maze=False, goal_set=None):
        self.env_fn = env
        env = self.env_fn()
        self.env = env
        self.n = env.num
        self.render_mode = render_mode
        self.visibility = visibility
        self.fixed_maze = fixed_maze
        self.goal_set = goal_set  # list of (x, y) goal positions to sample from

        # world_dim is determined after the first reset; placeholder 15 (easy)
        self._world_dim = 15
        self.observation_space = spaces.Box(
            -1, 1, (self._world_dim, self._world_dim, 3), np.float32,
        )
        self.action_space = spaces.Discrete(N_ACT)
        self._opt = [None] * self.n
        self._apos = [None] * self.n
        self._gpos = [None] * self.n
        self._last_rgb = [None] * self.n
        self._nav = [None] * self.n
        self._wd = [None] * self.n

        # fixed maze cache
        self._fixed_nav = None
        self._fixed_wd = None
        self._free_cells = None

    def _solve_all(self):
        for i in range(self.n):
            nav, apos, gpos, wd = _extract_grid(self.env, i)
            self._nav[i] = nav
            self._apos[i] = apos
            self._gpos[i] = gpos
            self._wd[i] = wd
            if gpos is not None:
                self._opt[i] = _q_iteration(nav, gpos, wd)
            else:
                self._opt[i] = None

    def _solve_one(self, i):
        nav, apos, gpos, wd = _extract_grid(self.env, i)
        self._nav[i] = nav
        self._apos[i] = apos
        self._gpos[i] = gpos
        self._wd[i] = wd
        if gpos is not None:
            self._opt[i] = _q_iteration(nav, gpos, wd)
        else:
            self._opt[i] = None

    def _init_fixed_maze(self):
        """Cache maze layout from seed 42."""
        tmp = ProcgenGym3Env(
            num=1, env_name="maze", start_level=42, num_levels=1,
            distribution_mode="easy", center_agent=True,
            restrict_themes=True, use_backgrounds=False,
            use_monochrome_assets=True,
        )
        nav, _, _, wd = _extract_grid(tmp, 0)
        self._fixed_nav = nav.copy()
        self._fixed_wd = wd
        self._world_dim = wd
        self.observation_space = spaces.Box(
            -1, 1, (wd, wd, 3), np.float32)
        self._free_cells = [
            (x, y) for y in range(wd) for x in range(wd) if nav[y, x]
        ]

    def _reset_sub_env(self, i):
        """Reset sub-env i with random start and goal from goal_set."""
        self._nav[i] = self._fixed_nav.copy()
        self._wd[i] = self._fixed_wd
        # sample goal from goal_set
        gpos = self.goal_set[np.random.randint(len(self.goal_set))]
        self._gpos[i] = gpos
        # sample start from free cells (not goal)
        free = [c for c in self._free_cells if c != gpos]
        self._apos[i] = free[np.random.randint(len(free))]
        self._opt[i] = _q_iteration(self._nav[i], gpos, self._fixed_wd)

    # ------------------------------------------------------------------
    #  Grid observation builder
    # ------------------------------------------------------------------
    def _build_grid_obs(self):
        """Build (world_dim, world_dim, 3) observations for all sub-envs.

        Channel 0 – wall:   1 = wall,  0 = free space, −1 = outside visibility
        Channel 1 – goal:   1 = goal,  0 = not goal,   −1 = outside visibility
        Channel 2 – agent:  1 = agent, 0 = empty,      −1 = outside visibility

        When self.visibility is None every cell is visible.
        Otherwise only a visibility × visibility window centred on the agent
        is revealed; all other cells are set to −1 in every channel.

        Returns np.ndarray of shape (n, wd, wd, 3) float32.
        """
        wd = self._world_dim
        obs = np.full((self.n, wd, wd, 3), -1.0, dtype=np.float32)

        for i in range(self.n):
            ax, ay = self._apos[i]
            nav = self._nav[i]
            gpos = self._gpos[i]
            gx, gy = gpos if gpos is not None else (-1, -1)

            # determine which cells are visible
            if self.visibility is not None:
                half = self.visibility // 2
                x_lo = max(0, ax - half)
                x_hi = min(wd, ax + half + 1)
                y_lo = max(0, ay - half)
                y_hi = min(wd, ay + half + 1)
            else:
                x_lo, x_hi = 0, wd
                y_lo, y_hi = 0, wd

            for y in range(y_lo, y_hi):
                for x in range(x_lo, x_hi):
                    # ch0: wall (1) / free (0)
                    obs[i, y, x, 0] = 0.0 if nav[y, x] else 1.0
                    # ch1: goal
                    obs[i, y, x, 1] = 1.0 if (x == gx and y == gy) else 0.0
                    # ch2: agent
                    obs[i, y, x, 2] = 1.0 if (x == ax and y == ay) else 0.0

        return obs

    def reset(self):
        self.env = self.env_fn()
        _, ob, _ = self.env.observe()

        if self.fixed_maze:
            if self._fixed_nav is None:
                self._init_fixed_maze()
            for i in range(self.n):
                self._reset_sub_env(i)
        else:
            self._solve_all()
            if self._wd[0] is not None and self._wd[0] != self._world_dim:
                self._world_dim = self._wd[0]
                self.observation_space = spaces.Box(
                    -1, 1, (self._world_dim, self._world_dim, 3), np.float32)

        raw_rgb = ob["rgb"]
        obs = self._build_grid_obs()
        infos = self._build_infos(raw_rgb=raw_rgb)
        return obs, infos

    def step(self, actions):
        if self.fixed_maze:
            # simulate movement ourselves
            rews = np.zeros(self.n, dtype=np.float32)
            dones = np.zeros(self.n, dtype=bool)
            for i in range(self.n):
                ax, ay = self._apos[i]
                a = int(actions[i])
                nx, ny = ax + DX[a], ay + DY[a]
                wd = self._fixed_wd
                if 0 <= nx < wd and 0 <= ny < wd and self._nav[i][ny, nx]:
                    self._apos[i] = (nx, ny)
                if self._apos[i] == self._gpos[i]:
                    rews[i] = 10.0
                    dones[i] = True
                    self._reset_sub_env(i)
            _, ob, _ = self.env.observe()
            raw_rgb = ob["rgb"]
            obs = self._build_grid_obs()
            infos = self._build_infos(raw_rgb=raw_rgb)
            return obs, rews, dones, infos

        pa = np.array(
            [ACT_MAP[int(a)] for a in actions], dtype=np.int32
        )
        self.env.act(pa)
        rew, ob, first = self.env.observe()

        raw_rgb = ob["rgb"]
        rews = rew.astype(np.float32)
        dones = first.astype(bool)

        # re-extract agent positions from true binary state
        for i in range(self.n):
            if dones[i]:
                self._solve_one(i)
            else:
                self._refresh_apos(i)

        obs = self._build_grid_obs()
        infos = self._build_infos(raw_rgb=raw_rgb)
        return obs, rews, dones, infos

    def _refresh_apos(self, i):
        """Re-read agent position from binary state."""
        nav, apos, _, _ = _extract_grid(self.env, i)
        self._apos[i] = apos

    def _build_infos(self, raw_rgb=None):
        raw = self.env.get_info()
        infos = []
        for i in range(self.n):
            d = {}
            # always include the full 64x64 RGB for video rendering
            if raw_rgb is not None:
                d["full_obs"] = raw_rgb[i]
            if self.render_mode == "rgb_array" and "rgb" in raw[i]:
                d["rgb"] = raw[i]["rgb"]
                self._last_rgb[i] = raw[i]["rgb"]
            if self._opt[i] is not None and self._apos[i] is not None:
                ax, ay = self._apos[i]
                oa = self._opt[i][ay, ax]
                d["opt_action"] = int(oa) if oa >= 0 else 0
            if self._nav[i] is not None and self._opt[i] is not None:
                d["opt_grid"] = _render_opt_grid(
                    self._nav[i], self._opt[i],
                    self._apos[i], self._gpos[i],
                    self._wd[i],
                )
            infos.append(d)
        return infos

    def render(self, idx=0):
        if self._last_rgb[idx] is not None:
            return self._last_rgb[idx]
        _, ob, _ = self.env.observe()
        return ob["rgb"][idx]


def _get_free_cells_seed42():
    """Extract free cells from maze at seed 42."""
    tmp = ProcgenGym3Env(
        num=1, env_name="maze", start_level=42, num_levels=1,
        distribution_mode="easy", center_agent=True,
        restrict_themes=True, use_backgrounds=False,
        use_monochrome_assets=True,
    )
    nav, _, _, wd = _extract_grid(tmp, 0)
    return [(x, y) for y in range(wd) for x in range(wd) if nav[y, x]]


def make_maze_envs(
    n_train=8,
    n_eval=20,
    train_start=0,
    train_levels=100,
    eval_start=100,
    eval_levels=20,
    visibility=None,
    fixed_maze=False,
    train_goal_ratio=0.8,
):
    """Create train and eval ProcgenMaze envs wrapped in gymnasium style.

    Parameters
    ----------
    visibility : int or None
        If set, observations are ``(world_dim, world_dim, 3)`` grids with
        partial visibility masking (-1 for unseen cells).
        See :class:`VecProcgenMaze`.
    fixed_maze : bool
        If True, use a single maze (seed 42) and only randomize start/goal.
    train_goal_ratio : float
        Fraction of free cells to use as train goals (rest for eval).

    Notes
    -----
    With ``fixed_maze=False`` (default), train/eval generalization split is
    internal to Procgen and controlled by ``train_start/train_levels`` and
    ``eval_start/eval_levels``. No explicit goal split is used in that mode.
    """
    # build goal sets if fixed_maze
    train_goals, eval_goals = None, None
    if fixed_maze:
        free = _get_free_cells_seed42()
        rng = np.random.default_rng(42)
        idxs = rng.permutation(len(free))
        n_train_goals = int(len(free) * train_goal_ratio)
        train_goals = [free[i] for i in idxs[:n_train_goals]]
        eval_goals = [free[i] for i in idxs[n_train_goals:]]
        if len(train_goals) == 0 or len(eval_goals) == 0:
            raise ValueError(
                "fixed_maze=True requires non-empty train/eval goal sets; "
                "adjust train_goal_ratio away from 0.0/1.0"
            )

    train_g3 = lambda: ProcgenGym3Env(
        num=n_train,
        env_name="maze",
        start_level=train_start,
        num_levels=train_levels,
        distribution_mode="easy",
        center_agent=True,
        restrict_themes=True,
        use_backgrounds=False,
        use_monochrome_assets=True,
    )
    eval_g3 = lambda: ProcgenGym3Env(
        num=n_eval,
        env_name="maze",
        start_level=eval_start,
        num_levels=eval_levels,
        distribution_mode="easy",
        center_agent=True,
        restrict_themes=True,
        use_backgrounds=False,
        render_mode="rgb_array",
        use_monochrome_assets=True,
    )
    train_env = VecProcgenMaze(train_g3, visibility=visibility,
                                fixed_maze=fixed_maze, goal_set=train_goals)
    eval_env = VecProcgenMaze(eval_g3, render_mode="rgb_array",
                              visibility=visibility,
                              fixed_maze=fixed_maze, goal_set=eval_goals)
    return train_env, eval_env


def _render_grid_obs(obs_3ch, cell_px=40):
    """Render a (world_dim, world_dim, 3) grid observation as an RGB image.

    Channel semantics (values -1 / 0 / 1):
      ch0 – walls:  1=wall, 0=free, -1=outside visibility
      ch1 – goal:   1=goal, 0=not goal, -1=outside visibility
      ch2 – agent:  1=agent, 0=empty, -1=outside visibility

    Colour mapping:
      * Gray (120) – outside visibility (any channel == -1)
      * Black      – wall  (ch0==1)
      * White      – free navigable cell
      * Green      – goal  (ch1==1)
      * Red        – agent (ch2==1)

    Returns uint8 array of shape (wd*cell_px, wd*cell_px, 3).
    """
    wd = obs_3ch.shape[0]
    size = wd * cell_px
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for gy in range(wd):
        for gx in range(wd):
            py = gy * cell_px
            px = gx * cell_px
            ch0, ch1, ch2 = obs_3ch[gy, gx]
            if ch0 < -0.5:
                # outside visibility window → gray
                img[py:py + cell_px, px:px + cell_px] = 120
            elif ch0 > 0.5:
                # wall → black (already 0)
                pass
            else:
                # free cell → white
                img[py:py + cell_px, px:px + cell_px] = 255
            # overlay goal (green)
            if ch1 > 0.5:
                img[py:py + cell_px, px:px + cell_px] = (0, 200, 0)
            # overlay agent (red, slightly inset)
            if ch2 > 0.5:
                m = cell_px // 5
                img[py + m:py + cell_px - m,
                    px + m:px + cell_px - m] = (220, 40, 40)
    # draw grid lines
    for k in range(wd + 1):
        coord = k * cell_px
        if coord < size:
            img[coord, :] = 80
            img[:, coord] = 80
    return img


def _rollout_multi(env, policy, n_episodes=5, max_t=500):
    """Run rollouts collecting episodes from all sub-envs.

    Returns dict mapping (env_idx, episode_number) to
    list of (bin_obs_rgb, full_obs, rgb, opt_grid) frames.
    *bin_obs_rgb* is the rendered binary obs (or None if not using visibility).
    """
    obs, infos = env.reset()
    ep_count = [0] * env.n
    cur_frames = [[] for _ in range(env.n)]
    all_episodes = {}

    for t in range(max_t * n_episodes):
        acts = policy(obs, infos, env.n)
        for i in range(env.n):
            rgb = infos[i].get("rgb", obs[i])
            og = infos[i].get("opt_grid", None)
            full_obs = infos[i].get("full_obs", obs[i])
            # render binary obs if visibility is on
            if env.visibility is not None:
                bin_panel = _render_grid_obs(obs[i])
            else:
                bin_panel = None
            cur_frames[i].append(
                (bin_panel, full_obs.copy(), rgb.copy(),
                 og.copy() if og is not None else None)
            )
        obs, rews, dones, infos = env.step(acts)
        for i in range(env.n):
            if dones[i]:
                if ep_count[i] < n_episodes:
                    all_episodes[(i, ep_count[i])] = cur_frames[i]
                ep_count[i] += 1
                cur_frames[i] = []
        if all(c >= n_episodes for c in ep_count):
            break
    for i in range(env.n):
        if ep_count[i] < n_episodes and cur_frames[i]:
            all_episodes[(i, ep_count[i])] = cur_frames[i]
    return all_episodes


def _save_video(frames, path, fps=10, labels=None, panel_size=256):
    """Save list of (bin_obs_rgb, full_obs, rgb, opt_grid) as side-by-side mp4.

    Panels (left to right): binary-obs | full-obs | procgen-render | opt-grid.
    Panels that are None are skipped.  All panels are resized to
    *panel_size × panel_size*.

    Parameters
    ----------
    labels : list[str] or None
        If provided, a text label is drawn on each frame (same length as
        *frames*).  Useful for showing "EXPERT" / "POLICY" per timestep.
    panel_size : int
        Height and width (in pixels) of each panel in the video.
    """
    import imageio_ffmpeg
    from PIL import ImageDraw
    ps = panel_size
    combined = []
    for idx, (bin_f, full_f, rgb_f, og_f) in enumerate(frames):
        panels = []
        if bin_f is not None:
            panels.append(np.array(
                Image.fromarray(bin_f).resize((ps, ps), Image.NEAREST)))
        if full_f is not None:
            panels.append(np.array(
                Image.fromarray(full_f).resize((ps, ps), Image.NEAREST)))
        if rgb_f is not None:
            panels.append(np.array(
                Image.fromarray(rgb_f).resize((ps, ps), Image.NEAREST)))
        if og_f is not None:
            panels.append(np.array(
                Image.fromarray(og_f).resize((ps, ps), Image.NEAREST)))
        row = np.concatenate(panels, axis=1)

        # draw text label if provided
        if labels is not None and idx < len(labels):
            pil_img = Image.fromarray(row)
            draw = ImageDraw.Draw(pil_img)
            txt = labels[idx]
            color = (0, 255, 0) if "EXPERT" in txt else (255, 100, 100)
            draw.text((5, 5), txt, fill=color)
            row = np.array(pil_img)

        combined.append(row)
    h, w = combined[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        path, (w, h), fps=fps, pix_fmt_in="rgb24"
    )
    writer.send(None)
    for frame in combined:
        writer.send(frame.tobytes())
    writer.close()
    print(f"saved {path} ({len(frames)} frames)")


if __name__ == "__main__":
    import os

    out_dir = "maze_debug_videos"
    os.makedirs(out_dir, exist_ok=True)

    n_envs = 4
    n_episodes = 3
    _, eval_env = make_maze_envs(
        n_train=2, n_eval=n_envs,
        train_levels=10, eval_levels=20,
        visibility=7,
    )

    obs, infos = eval_env.reset()
    print(f"obs shape: {obs.shape}, obs dtype: {obs.dtype}")
    print(f"info keys: {list(infos[0].keys())}")
    if "rgb" in infos[0]:
        print(f"info rgb shape: {infos[0]['rgb'].shape}")
    if "full_obs" in infos[0]:
        print(f"info full_obs shape: {infos[0]['full_obs'].shape}")
    print(f"opt_action: {infos[0].get('opt_action', 'N/A')}")

    # save static images for each env
    for i in range(n_envs):
        og = infos[i].get("opt_grid")
        if og is not None:
            p = os.path.join(out_dir, f"opt_grid_env{i}.png")
            Image.fromarray(og).save(p)
            print(f"saved {p}")
        # render and save the binary obs
        bin_img = _render_grid_obs(obs[i])
        p = os.path.join(out_dir, f"grid_obs_env{i}.png")
        Image.fromarray(bin_img).save(p)
        print(f"saved {p}")
        # print agent/goal positions for debugging
        print(f"  env{i}: agent={eval_env._apos[i]}, "
              f"goal={eval_env._gpos[i]}, "
              f"world_dim={eval_env._wd[i]}")

    # --- expert rollouts ---
    def expert_policy(obs, infos, n):
        return np.array(
            [inf.get("opt_action", 0) for inf in infos]
        )

    print(f"\nrunning expert rollouts ({n_episodes} eps "
          f"x {n_envs} envs)...")
    episodes = _rollout_multi(
        eval_env, expert_policy, n_episodes=n_episodes
    )
    for (ei, ep), frames in sorted(episodes.items()):
        path = os.path.join(
            out_dir, f"expert_env{ei}_ep{ep}.mp4"
        )
        _save_video(frames, path)

    # --- random rollouts ---
    def random_policy(obs, infos, n):
        return np.array(
            [np.random.randint(N_ACT) for _ in range(n)]
        )

    print(f"\nrunning random rollouts ({n_episodes} eps "
          f"x {n_envs} envs)...")
    episodes = _rollout_multi(
        eval_env, random_policy, n_episodes=n_episodes
    )
    for (ei, ep), frames in sorted(episodes.items()):
        path = os.path.join(
            out_dir, f"random_env{ei}_ep{ep}.mp4"
        )
        _save_video(frames, path)

    print(f"\ndone. videos saved to {out_dir}/")
