import numpy as np
from stable_baselines3 import DQN

from envs.lane_avoid_env import LaneAvoidEnv


def safety_shield(action: int, obs: np.ndarray) -> int:
    """
    Jednostavan safety shield:
    - gleda najbližu prepreku (dx1, dy1) iz observationa
    - ako je prepreka jako blizu (dx < 0.6) i skoro u istoj liniji (|dy| < 0.25),
      ne dopušta akciju koja ide prema toj strani
    """
    y = float(obs[0])
    dx1 = float(obs[2])
    dy1 = float(obs[3])

    # ako je prepreka blizu ispred i u sličnoj bočnoj poziciji (opasnost od sudara)
    if dx1 < 0.9 and abs((y - dy1)) < 0.35:
        # ako je prepreka "lijevo" od auta, ne idi lijevo
        if dy1 < y and action == 0:
            return 1  # ravno
        # ako je prepreka "desno" od auta, ne idi desno
        if dy1 > y and action == 2:
            return 1  # ravno

        # ako su skoro skroz poravnati, najbolje je "ravno" (stabilno)
        if abs(dy1 - y) < 0.18:
            return 1

    return action


def run_episode(env, model=None, use_shield=False):
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
            action = int(action)

        if use_shield:
            action = safety_shield(action, obs)

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

    settings = [
        ("RANDOM", None, False),
        ("DQN", model, False),
        ("DQN + SHIELD", model, True),
    ]

    for label, pol, shield in settings:
        rewards, steps, events = [], [], []
        for _ in range(100):
            r, s, e = run_episode(env, model=pol, use_shield=shield)
            rewards.append(r)
            steps.append(s)
            events.append(e)
        summarize(label, rewards, steps, events)


if __name__ == "__main__":
    main()
