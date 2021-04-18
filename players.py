import random
from math import inf
import numpy as np
from chessbot_utils import fast_predict
from math import inf
import chess
import chess.engine
import cProfile, pstats


class Player:
    def __init__(
        self, team: int, encoder: dict, score_encoder: dict, boosted_rewards: bool
    ):
        self.team = 1 if team == 1 else -1
        self.encoder = encoder
        self.score_encoder = score_encoder
        self.boosted_rewards = boosted_rewards
        self.history = []

    def generate_move(self, env):
        starting_board = self.encode_env(env)
        action = random.sample(list(env.legal_moves), 1)[0]
        env.push(action)
        self.history.append([starting_board, self.encode_env(env)])
        return action

    def encode_env(self, env) -> np.array:
        temp = [self.encoder[x] for x in str(env).split()]
        temp.append([0, 0, 0, 0, 0, 0, self.team])
        return temp

    def get_board_score(self, encoded_board, team) -> float:  # returns piece score
        score = 0
        # if team == 1:
        #     for i in range(encoded_board.shape[0] - 1):
        #         val = self.score_encoder[tuple(encoded_board[i])]
        #         if val > 0:
        #             score += abs(val)
        # else:
        #     for i in range(encoded_board.shape[0] - 1):
        #         val = self.score_encoder[tuple(encoded_board[i])]
        #         if val < 0:
        #             score += abs(val)
        if team == 1:
            for i in range(len(encoded_board)-1):
                val = self.score_encoder[tuple(encoded_board[i])]
                if val > 0:
                    score += abs(val)
        else:
            for i in range(len(encoded_board)-1):
                val = self.score_encoder[tuple(encoded_board[i])]
                if val < 0:
                    score += abs(val)

        # print(score)
        return score

    def finalize_data(self, reward: float, draw=False) -> list:
        labeled_data = []
        base_reward = 1
        enemy_team = 1 if self.team != 1 else -1
        if not draw:
            increment = reward / len(self.history)
            for i in range(len(self.history)):
                turn_reward = -inf
                base_reward = (i + 1) * increment
                if self.boosted_rewards and i < len(self.history) - 3:
                    before = self.get_board_score(self.history[i][0], enemy_team)
                    after = self.get_board_score(self.history[i][1], enemy_team)
                    if after < before:
                        turn_reward = before - after

                if turn_reward > base_reward:
                    labeled_data.append([self.history[i][1], turn_reward])
                else:
                    labeled_data.append([self.history[i][1], base_reward])
        else:
            my_score = self.get_board_score(self.history[-1][1], self.team)
            enemy_score = self.get_board_score(self.history[-1][1], self.team * -1)

            if my_score > enemy_score:
                labeled_data.append([self.history[-1][1], reward])

        # print(f"labeled data length: {len(labeled_data)}")
        # print(f"labeled data[0] length: {len(labeled_data[0])}")
        # print(f"labeled data[0][0] length: {len(labeled_data[0][0])}")
        # print(f"labeled data[0][1] : {labeled_data[0][1]}")
        # print()


        return labeled_data

    def quit(self):
        pass

    def __repr__(self):
        return f"Random mover {'WHITE' if self.team == 1 else 'BLACK'} "


class chess_engine(Player):
    def __init__(
        self,
        team: int,
        encoder: dict,
        score_encoder: dict,
        boosted_rewards: bool,
        engine_path: str,
        time_limit: float,
        engine_name: str,
    ):
        super().__init__(team, encoder, score_encoder, boosted_rewards)
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.time_limit = time_limit
        self.name = engine_name

    def __repr__(self):
        return f"{self.name} chess engine {'BLACK' if self.team == -1 else 'WHITE'}"

    def generate_move(self, env):
        starting_board = self.encode_env(env)
        action = self.engine.play(env, chess.engine.Limit(time=self.time_limit)).move
        env.push(action)
        self.history.append([starting_board, self.encode_env(env)])
        return action

    def quit(self):
        self.engine.quit()


class ML_Player(Player):
    def __init__(
        self,
        team: int,
        encoder: dict,
        score_encoder: dict,
        boosted_rewards: bool,
        model,
        epsilon=0.95,
    ):
        super().__init__(team, encoder, score_encoder, boosted_rewards)
        self.model = model
        self.epsilon = epsilon

    def generate_move(self, env):
        starting_board = self.encode_env(env)
        if random.random() < self.epsilon:
            best_move = None
            best_score = -inf
            chosen_board = None

            #print("==========================")
            for count, action in enumerate(env.legal_moves):
                same = True
                env.push(action)
                encoded_board = self.encode_env(env)
                pred_board = np.array([encoded_board])
                # print(f"encoded board shape: {encoded_board.shape}")
                pred = float(fast_predict(pred_board, self.model))
                #print(pred)
                env.pop()
                if pred > best_score:
                    best_move = action
                    best_score = pred
                    chosen_board = encoded_board

        else:
            best_move = random.sample(list(env.legal_moves), 1)[0]
            env.push(best_move)
            chosen_board = self.encode_env(env)

        self.history.append([starting_board, chosen_board])
        return best_move

    def __repr__(self):
        return f"ML bot {'WHITE' if self.team == 1 else 'BLACK'} "


class ML_Player_Trainer(ML_Player):
    def __init__(
        self,
        team: int,
        encoder: dict,
        score_encoder: dict,
        boosted_rewards: bool,
        model,
        epsilon=1,
    ):
        super().__init__(team, encoder, score_encoder, boosted_rewards, model, epsilon)

    def __repr__(self):
        return f"ML trainer bot {'WHITE' if self.team == 1 else 'BLACK'} "


class ML_Distiller(ML_Player):
    def __init__(
        self,
        team: int,
        encoder: dict,
        score_encoder: dict,
        boosted_rewards: bool,
        model,
        epsilon=1,
        depth=2,
        speed_hacks=True,
    ):
        super().__init__(team, encoder, score_encoder, boosted_rewards, model, epsilon)
        self.depth = depth
        self.speed_hacks = speed_hacks
        self.speed_barrier = 15
        # self.profiler = cProfile.Profile()

    def encode_env(self, env, team) -> np.array:
        temp = [self.encoder[x] for x in str(env).split()]
        temp.append([0, 0, 0, 0, 0, 0, team])
        return np.array(temp)

    def recursive_move_selector(self, env, team, legal_moves, depth):
        _env = env.copy(stack=False)
        starting_board = self.encode_env(_env, self.team)
        if depth <= 1:
            temp = self._generate_move(_env, team)
            return temp[1], temp[0]
        best_score = -999999
        best_move = None

        for move in legal_moves:
            _env.push(move)
            encoded_board = self.encode_env(_env, team)
            pred_board = np.array([encoded_board])
            pred = float(fast_predict(pred_board, self.model))
            pred += self.terminal_balancer(_env, encoded_board, team)
            enemy_move = self._generate_move(_env, team * -1)[0]
            if enemy_move != None:
                _env.push(enemy_move)
                pred += self.recursive_move_selector(
                    _env, team, list(_env.legal_moves), depth - 1
                )[0]
                _env.pop()
            _env.pop()
            if pred > best_score:
                best_score = pred
                best_move = move

        if best_move != None:
            return best_score, best_move
        else:
            return 0, legal_moves[0] if len(legal_moves) > 0 else "xyz"

    def terminal_balancer(self, env, encoded_board, team) -> float:
        legal_moves = list(env.legal_moves)
        result = env.result()
        if result != "*":
            if result == "1-0" and team == 1:
                return 1

            elif result == "0-1" and team == -1:
                return 1

            elif result == "1/2-1/2":
                enemy_score = self.get_board_score(
                    encoded_board, 1 if team != 1 else -1
                )
                my_score = self.get_board_score(encoded_board, team)
                ratio = my_score / enemy_score
                return 0.5 if ratio <= 0.75 else -0.5

            else:
                if env.is_game_over():
                    print("result and game over mismatch!")
                    return None
                else:
                    return None

        else:
            for action in legal_moves:
                env.push(action)
                result = env.result()
                env.pop()
                if result == "1-0" and team != 1:
                    return -1

                elif result == "0-1" and team != -1:
                    return -1

        return 0

    def generate_move(self, env):
        starting_board = self.encode_env(env, self.team)
        legal_moves = list(env.legal_moves)
        if len(legal_moves) > 1:
            if random.random() < self.epsilon:
                # self.profiler.enable()
                if (
                    not self.speed_hacks
                    or self.depth == 1
                    or len(self.history) < self.speed_barrier
                ):
                    ideal_move = self.recursive_move_selector(
                        env, self.team, legal_moves, 1
                    )[1]
                else:
                    ideal_move = self.recursive_move_selector(
                        env, self.team, legal_moves, self.depth
                    )[1]
                # self.profiler.disable()
                # stats = pstats.Stats(self.profiler).sort_stats('cumtime')
                # stats.print_stats()
                if type(ideal_move) == str:
                    ideal_move = random.sample(legal_moves, 1)[0]
                    print(f"Had to pick random move! {len(legal_moves)}")

            else:
                ideal_move = random.sample(list(env.legal_moves), 1)[0]

        else:
            ideal_move = legal_moves[0]
        env.push(ideal_move)
        chosen_board = self.encode_env(env, self.team)

        self.history.append([starting_board, chosen_board])
        return ideal_move

    def _generate_move(self, env, team):
        best_move = None
        best_score = -inf

        for action in env.legal_moves:
            env.push(action)
            encoded_board = self.encode_env(env, team)
            pred = float(fast_predict(encoded_board, self.model))
            # print(pred)
            pred += self.terminal_balancer(env, encoded_board, team)
            env.pop()
            if pred > best_score:
                best_move = action
                best_score = pred
        return best_move, best_score

    def __repr__(self):
        return f"Distiller bot {'WHITE' if self.team == 1 else 'BLACK'} "


if __name__ == "__main__":
    pass
