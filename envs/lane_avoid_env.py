import gymnasium as gym
from gymnasium import spaces
import numpy as np


class LaneAvoidEnv(gym.Env):
    """
    Jednostavno 2D okruženje:
    - cesta je y u rasponu [-1, 1]
    - auto ima bočni položaj y i bočnu brzinu vy
    - prepreke su ispred i "dolaze" prema autu
    - akcije: 0 lijevo, 1 ravno, 2 desno
    Observation: [y, vy, dx1, dy1, dx2, dy2]
    """

    def __init__(self, dt: float = 0.1, num_obstacles: int = 2, max_steps: int = 300):
        super().__init__()

        self.dt = dt
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps

        # stanje auta
        self.y = 0.0
        self.vy = 0.0

        # prepreke: lista [dx, dy]
        self.obstacles = []

        self.step_count = 0

        # Akcije: 0 lijevo, 1 ravno, 2 desno
        self.action_space = spaces.Discrete(3)

        # Observation: y, vy, (dx,dy)*num_obstacles
        # Definiramo realne granice radi stabilnosti u učenju
        high = np.array([1.0, 1.0] + [5.0, 2.0] * self.num_obstacles, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        self.y = 0.0
        self.vy = 0.0

        self.obstacles = []
        for _ in range(self.num_obstacles):
            dx = float(self.np_random.uniform(2.0, 5.0))
            dy = float(self.np_random.uniform(-0.8, 0.8))
            self.obstacles.append([dx, dy])

        return self._get_obs(), {}

    def step(self, action: int):
        self.step_count += 1

        # 1) akcija utječe na bočnu brzinu
        if action == 0:      # lijevo
            self.vy -= 0.5
        elif action == 2:    # desno
            self.vy += 0.5
        # action == 1 -> ravno (bez promjene)

        # ograniči vy
        self.vy = float(np.clip(self.vy, -1.0, 1.0))

        # 2) update položaja auta
        self.y += self.vy * self.dt

        # 3) pomak prepreka prema autu (dx se smanjuje)
        for obs in self.obstacles:
            obs[0] -= 0.25  # "relativna brzina" prepreka

        # 4) respawn prepreka kad prođu auto
        for obs in self.obstacles:
            if obs[0] < -0.5:
                obs[0] = float(self.np_random.uniform(3.0, 5.0))
                obs[1] = float(self.np_random.uniform(-0.8, 0.8))

        terminated = False
        truncated = False

        # osnovna nagrada: "živi dulje"
        reward = 0.05

        # 5) izlazak iz trake
        if abs(self.y) > 1.0:
            terminated = True
            reward = -5.0

        # 6) sudar: ako je prepreka vrlo blizu u x i y
        if not terminated:
            for dx, dy in self.obstacles:
                if abs(dx) < 0.2 and abs(self.y - dy) < 0.2:
                    terminated = True
                    reward = -10.0
                    break

        # 7) kazne za "nesigurnu" vožnju (glatko + centrirano)
        reward -= 0.10 * abs(self.y)
        reward -= 0.05 * abs(self.vy)

        # 8) truncation ako epizoda traje predugo
        if self.step_count >= self.max_steps and not terminated:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, {}

    def _get_obs(self):
        obs = [self.y, self.vy]
        for dx, dy in self.obstacles:
            obs.extend([dx, dy])
        return np.array(obs, dtype=np.float32)
