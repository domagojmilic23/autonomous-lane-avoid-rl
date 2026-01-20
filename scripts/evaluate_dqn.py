import numpy as np
from stable_baselines3 import DQN

from envs.lane_avoid_env import LaneAvoidEnv


def run_episode(env, model=None):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0
    steps = 0

    while not (terminated or truncated):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        steps += 1

    return total_reward, steps


def main():
    env = LaneAvoidEnv()

    # Random baseline
    random_rewards, random_steps = [], []
    for _ in range(50):
        r, s = run_episode(env, model=None)
        random_rewards.append(r)
        random_steps.append(s)

    # DQN trained
    model = DQN.load("results/dqn_lane_avoid")
    dqn_rewards, dqn_steps = [], []
    for _ in range(50):
        r, s = run_episode(env, model=model)
        dqn_rewards.append(r)
        dqn_steps.append(s)

    print("RANDOM: prosj. reward =", float(np.mean(random_rewards)), "| prosj. koraci =", float(np.mean(random_steps)))
    print("DQN:    prosj. reward =", float(np.mean(dqn_rewards)), "| prosj. koraci =", float(np.mean(dqn_steps)))


if __name__ == "__main__":
    main()
