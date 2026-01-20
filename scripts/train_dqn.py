import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from envs.lane_avoid_env import LaneAvoidEnv


def main():
    # osiguraj folder za rezultate
    os.makedirs("results", exist_ok=True)

    # okruženje + monitor (sprema statistike epizoda)
    env = LaneAvoidEnv()
    env = Monitor(env, filename="results/monitor.csv")

    # DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=2_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
    )

    # treniraj (brza, ali korisna prva verzija)
    model.learn(total_timesteps=50_000)

    # spremi model
    model.save("results/dqn_lane_avoid")

    env.close()
    print("Trening gotov. Model spremljen u results/dqn_lane_avoid.zip")


if __name__ == "__main__":
    main()
