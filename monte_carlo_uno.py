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
            [self.deck.pop() for _ in range(5)]
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
        # First-visit Monte Carlo
        G = final_reward
        returns = []

        for state, action in reversed(episode):
            returns.append((state, action, G))
            G *= self.gamma

        visited = set()

        for state, action, G in reversed(returns):
            key = (state, action)

            if key in visited:
                continue

            visited.add(key)

            self.N[key] += 1
            self.Q[key] += (
                G - self.Q[key]
            ) / self.N[key]


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
            agent.update(episode, reward)
            return winner

    return None


def main():
    num_games = 10000

    game = Uno()
    agent = MonteCarloAgent(
        epsilon=0.1,
        gamma=0.9
    )

    player1_wins = 0
    player2_wins = 0
    draws = 0

    print("Training Monte Carlo UNO Agent.")

    for i in range(num_games):
        winner = play_game(game, agent)

        if winner == 0:
            player1_wins += 1
        elif winner == 1:
            player2_wins += 1
        else:
            draws += 1

        if (i + 1) % 1000 == 0:
            print(f"\nGames Played: {i + 1}")
            print(f"Player 1 Wins: {player1_wins}")
            print(f"Player 2 Wins: {player2_wins}")
            print(f"Draws: {draws}")
            print(
                f"Player 1 Win Rate: "
                f"{player1_wins / (i + 1):.2%}"
            )

    print("\nTraining Complete!")
    print(f"Total Games: {num_games}")
    print(f"Player 1 Wins: {player1_wins}")
    print(f"Player 2 Wins: {player2_wins}")
    print(f"Draws: {draws}")
    print(
        f"Final Player 1 Win Rate: "
        f"{player1_wins / num_games:.2%}"
    )
    print(f"Learned State-Action Pairs: {len(agent.Q)}")


if __name__ == "__main__":
    main()