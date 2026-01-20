from envs.lane_avoid_env import LaneAvoidEnv

env = LaneAvoidEnv()
obs, _ = env.reset()

terminated = False
truncated = False
korak = 0

while not (terminated or truncated) and korak < 200:
    akcija = env.action_space.sample()
    obs, nagrada, terminated, truncated, _ = env.step(akcija)

    y = obs[0]
    vy = obs[1]
    print(f"Korak {korak:03d} | akcija={akcija} | y={y:.2f} | vy={vy:.2f} | nagrada={nagrada:.2f}")
    korak += 1

print("Epizoda završila.", "terminated" if terminated else "truncated")
