import numpy as np
from stable_baselines3 import DQN
from envs.lane_avoid_env import LaneAvoidEnv


def run_episode(env, model=None):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0
    steps = 0
    event = "none"

    while not (terminated or truncated):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        steps += 1
        event = info.get("event", "none")

    return total_reward, steps, event


def summarize(name, rewards, steps, events):
    print(f"\n=== {name} ===")
    print("Prosj. reward:", float(np.mean(rewards)))
    print("Prosj. koraci:", float(np.mean(steps)))

    total = len(events)
    collisions = sum(e == "collision" for e in events)
    lane_dep = sum(e == "lane_departure" for e in events)
    timeout = sum(e == "timeout" for e in events)

    print("Sudar (collision):", collisions, f"({collisions/total:.1%})")
    print("Izlazak iz trake (lane_departure):", lane_dep, f"({lane_dep/total:.1%})")
    print("Istek vremena (timeout):", timeout, f"({timeout/total:.1%})")


def main():
    env = LaneAvoidEnv()
    model = DQN.load("results/dqn_lane_avoid")

    for label, pol in [("RANDOM", None), ("DQN", model)]:
        rewards, steps, events = [], [], []
        for _ in range(100):
            r, s, e = run_episode(env, model=pol)
            rewards.append(r)
            steps.append(s)
            events.append(e)
        summarize(label, rewards, steps, events)


if __name__ == "__main__":
    main()
