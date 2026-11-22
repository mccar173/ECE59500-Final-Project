import random
from collections import defaultdict

COLORS = ["R", "G", "B", "Y"]


class Uno:
    def __init__(self):
        self.reset()

    def reset(self):
        self.deck = [
            (color, number)
            for color in COLORS
            for number in range(10)
        ] * 2

        random.shuffle(self.deck)

        self.hands = [
            [self.deck.pop() for _ in range(6)]
            for _ in range(2)
        ]

        self.top = self.deck.pop()
        self.player = 0

    def legal_actions(self, player):
        actions = [
            card for card in set(self.hands[player])
            if card[0] == self.top[0]
            or card[1] == self.top[1]
        ]

        return actions if actions else ["DRAW"]

    def step(self, action):
        p = self.player

        if action == "DRAW":
            if self.deck:
                self.hands[p].append(self.deck.pop())
        else:
            self.hands[p].remove(action)
            self.top = action

        if not self.hands[p]:
            return True, p

        self.player = 1 - p
        return False, None


class MonteCarloAgent:
    def __init__(self, epsilon=0.1, gamma=0.9):
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self.epsilon = epsilon
        self.gamma = gamma

    def choose_action(self, state, actions):
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.choice(actions)

        best_value = max(
            self.Q[(state, action)] for action in actions
        )

        best_actions = [
            action for action in actions
            if self.Q[(state, action)] == best_value
        ]

        return random.choice(best_actions)

    def update(self, episode, final_reward):
        G = final_reward
        returns = []

        for state, action in reversed(episode):
            returns.append((state, action, G))
            G *= self.gamma

        visited = set()
        q_changes = []

        for state, action, G in reversed(returns):
            key = (state, action)

            if key in visited:
                continue

            visited.add(key)

            old_q = self.Q[key]

            self.N[key] += 1
            self.Q[key] += (
                G - self.Q[key]
            ) / self.N[key]

            q_changes.append(abs(self.Q[key] - old_q))

        return sum(q_changes) / len(q_changes) if q_changes else 0


def get_state(game):
    return (
        tuple(sorted(game.hands[0])),
        game.top,
        len(game.hands[1])
    )


def play_game(game, agent, max_turns=200):
    game.reset()
    episode = []

    for _ in range(max_turns):
        player = game.player
        actions = game.legal_actions(player)

        if player == 0:
            state = get_state(game)
            action = agent.choose_action(state, actions)
            episode.append((state, action))
        else:
            action = random.choice(actions)

        done, winner = game.step(action)

        if done:
            reward = 1 if winner == 0 else -1
            q_change = agent.update(episode, reward)
            return winner, q_change

    return None, 0


import time
import numpy as np
import matplotlib.pyplot as plt


def main():
    num_games = 30000
    window = 500
    tolerance = 0.02
    patience = 5

    game = Uno()
    agent = MonteCarloAgent(
        epsilon=0.1,
        gamma=0.9
    )

    player1_wins = 0
    player2_wins = 0
    draws = 0

    win_history = []
    q_change_history = []

    win_rates = []
    q_changes = []
    episodes = []

    convergence_episode = None
    stable_count = 0

    start_time = time.perf_counter()

    print("Training Monte Carlo UNO Agent.")

    for i in range(num_games):
        winner, q_change = play_game(game, agent)

        if winner == 0:
            player1_wins += 1
            win_history.append(1)
        elif winner == 1:
            player2_wins += 1
            win_history.append(0)
        else:
            draws += 1
            win_history.append(0)

        q_change_history.append(q_change)

        if (i + 1) % window == 0:

            recent_win_rate = np.mean(
                win_history[-window:]
            )

            recent_q_change = np.mean(
                q_change_history[-window:]
            )

            win_rates.append(recent_win_rate)
            q_changes.append(recent_q_change)
            episodes.append(i + 1)

            # Check convergence using win rate
            if len(win_rates) >= 2:
                difference = abs(
                    win_rates[-1] - win_rates[-2]
                )

                if difference < tolerance:
                    stable_count += 1
                else:
                    stable_count = 0

                if (
                    stable_count >= patience
                    and convergence_episode is None
                ):
                    convergence_episode = i + 1

            print(
                f"Episode {i + 1}: "
                f"Win Rate = {recent_win_rate:.3f}, "
                f"Q Change = {recent_q_change:.6f}"
            )

    elapsed_time = time.perf_counter() - start_time

    print("\nTraining Complete!")
    print(f"Total Games: {num_games}")
    print(f"Player 1 Wins: {player1_wins}")
    print(f"Player 2 Wins: {player2_wins}")
    print(f"Draws: {draws}")

    print(
        f"Final Win Rate: "
        f"{player1_wins / num_games:.2%}"
    )

    print(
        f"Training Time: {elapsed_time:.2f} seconds"
    )

    print(
        f"Training Speed: "
        f"{num_games / elapsed_time:.2f} games/second"
    )

    if convergence_episode is not None:
        print(
            f"Convergence Episode: "
            f"{convergence_episode}"
        )
    else:
        print("Convergence not detected.")

    print(
        f"Learned State-Action Pairs: {len(agent.Q)}"
    )

    # Plot convergence
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(
        episodes,
        win_rates,
        label="Win Rate"
    )

    if convergence_episode is not None:
        axes[0].axvline(
            convergence_episode,
            linestyle="--",
            color="red",
            label="Convergence"
        )

    axes[0].set_xlabel("Training Episodes")
    axes[0].set_ylabel("Win Rate")
    axes[0].set_title("Monte Carlo UNO Win Rate Convergence")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        episodes,
        q_changes,
        label="Average Q-value Change"
    )

    axes[1].set_xlabel("Training Episodes")
    axes[1].set_ylabel("Average Absolute Q Change")
    axes[1].set_title("Monte Carlo Q-value Convergence")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()