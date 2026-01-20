import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Monitor CSV ima komentar linije na početku koje počinju s '#'
    df = pd.read_csv("results/monitor.csv", comment="#")

    # Kolone su tipično: r (reward), l (length), t (time)
    rewards = df["r"].to_numpy()
    lengths = df["l"].to_numpy()

    # Moving average radi ljepšeg grafa
    window = 20
    if len(rewards) >= window:
        ma = pd.Series(rewards).rolling(window).mean().to_numpy()
    else:
        ma = rewards

    plt.figure()
    plt.plot(rewards)
    plt.title("Nagrada po epizodi (reward)")
    plt.xlabel("Epizoda")
    plt.ylabel("Reward")
    plt.show()

    plt.figure()
    plt.plot(lengths)
    plt.title("Duljina epizode (broj koraka)")
    plt.xlabel("Epizoda")
    plt.ylabel("Koraci")
    plt.show()

    plt.figure()
    plt.plot(ma)
    plt.title(f"Moving average reward (window={window})")
    plt.xlabel("Epizoda")
    plt.ylabel("Reward (MA)")
    plt.show()


if __name__ == "__main__":
    main()
