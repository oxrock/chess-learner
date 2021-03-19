import random
from math import inf
import numpy as np

class Player:
    def __init__(self, team: int, encoder: dict):
        self.team = 1 if team == 1 else -1
        self.encoder = encoder
        self.history = []

    def generate_move(self, env):
        action = random.sample(list(env.legal_moves), 1)[0]
        env.push(action)
        self.history.append(self.encode_env(env))
        return action

    def encode_env(self, env) -> list:
        temp = [self.encoder[x] for x in str(env).split()]
        temp.append(self.team)
        return np.array([temp])

    def finalize_data(self, reward: int) -> list:
        reward_chunk = reward / len(self.history)
        labeled_data = []
        for i in range(len(self.history)):
            labeled_data.append([self.history[i], reward_chunk * (i + 1)])

        return labeled_data

    def __repr__(self):
        return f"Random mover {'-WHITE' if self.team == 1 else '-BLACK'} "

class ML_Player(Player):
    def __init__(self, team: int, encoder: dict,model,epsilon=0.9):
        super().__init__(team, encoder)
        self.model = model
        self.epsilon = epsilon


    def generate_move(self, env):
        if random.random() < self.epsilon:
            best_move = None
            best_score = -inf
            chosen_board = None

            for action in env.legal_moves:
                env.push(action)
                encoded_board = self.encode_env(env)
                #print(encoded_board.shape)
                pred = self.model.predict(encoded_board)[0][0]
                env.pop()
                if pred > best_score:
                    best_move = action
                    best_score = pred
                    chosen_board = encoded_board

        else:
            best_move = random.sample(list(env.legal_moves), 1)[0]
            env.push(best_move)
            chosen_board = self.encode_env(env)

        self.history.append(chosen_board)
        return best_move

    def __repr__(self):
        return f"ML bot {'-WHITE' if self.team == 1 else '-BLACK'} "

if __name__ == "__main__":
    pass