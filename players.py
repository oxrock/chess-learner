import random
import numpy as np
from chessbot_utils import fast_predict, create_piece_decoder, flatten_board
from math import inf
import chess
import chess.engine
# import cProfile
# import pstats




class Player:
    def __init__(
            self, team: int, encoder: dict, score_encoder: dict, boosted_rewards: bool
    ):
        self.team = 1 if team == 1 else -1
        self.encoder = encoder
        self.score_encoder = score_encoder
        self.decoder = create_piece_decoder()
        self.boosted_rewards = boosted_rewards
        self.gamma = np.exp(np.log(0.5) / 10)
        self.history = []
        self.ignore_types = [".", "1", "-1"]
        self.pieces = ["P", "N", "B", "R", "Q", "K"]
        if self.team != 1:
            self.pieces = [piece.lower() for piece in self.pieces]

    def generate_move(self, env):
        starting_board = self.encode_env(env, self.team)
        action = random.sample(list(env.legal_moves), 1)[0]
        env.push(action)
        chosen_board = self.encode_env(env, self.team)
        #print(chosen_board[-1])
        self.history.append([starting_board, chosen_board, action])
        return action

    def convert_board(self, board: list):
        _board = list(reversed(board))
        for i in range(len(_board)):
            if _board[i] == _board[i].upper():
                _board[i] = _board[i].lower()
            elif _board[i] == _board[i].lower():
                _board[i] = _board[i].upper()
        return _board



    def encode_env(self, env, team) -> list:
        env_list= str(env).split()
        if team != 1:
            env_list = self.convert_board(env_list)
        temp = [self.encoder[x] for x in env_list]
        _scores = self.get_board_score(temp, team)
        temp.append([float(_scores[0])/float(_scores[1])])
        return temp

    def repeat_move_adjuster(self, action, past_index) -> float:
        if len(self.history) < abs(past_index):
            return 0

        if past_index == -2:
            if self.history[past_index][2] == action:
                return -0.3
            return 0

        else:
            result = 0
            if self.history[past_index][2] == action:
                result -= (0.3 / abs(past_index))
            return result + self.repeat_move_adjuster(action, past_index + 1)

    def get_board_score(self, encoded_board, team, reverse=False) -> tuple:  # returns piece score per player
        my_score = 0
        enemy_score = 0
        remainder = len(encoded_board)-64
        if team == self.team:
            scan_team = 1
        else:
            scan_team = -1

        if reverse:
            scan_team *= -1

        if scan_team == 1:
            for i in range(len(encoded_board)-remainder):
                _score = self.score_encoder[tuple(encoded_board[i])]
                if abs(_score) > 0:
                    if _score > 0:
                        my_score += abs(_score)
                    else:
                        enemy_score += abs(_score)
        else:
            for i in range(len(encoded_board)-remainder):
                _score = self.score_encoder[tuple(encoded_board[i])]
                if abs(_score) > 0:
                    if _score < 0:
                        my_score += abs(_score)
                    else:
                        enemy_score += abs(_score)


        return my_score, enemy_score

    def get_enemy_greedy_move(self, env):
        starting_board = self.encode_env(env, self.team)
        player_score = self.get_board_score(starting_board, -self.team, reverse=True)[1]

        best_score = -inf
        best_move = None
        #print("===================")
        for action in env.legal_moves:
            env.push(action)
            encoded_board = self.encode_env(env, self.team*-1)
            pred = player_score - self.get_board_score(starting_board, -self.team, reverse=True)[1]
            if pred != 0:
                print(f"{player_score=}, {pred=}")
            result = self.terminal_balancer(env, encoded_board, self.team * -1, reverse=True)
            if result >= 1:
                pred += 100
            elif result <= -0.5:
                pred -= 10
            env.pop()

            if pred > best_score:
                best_score = pred
                best_move = action

        return best_score, best_move

    def move_reward_generator(self, index: int) -> float:
        if index < len(self.history) - 1:
            my_starting_points, enemy_starting_points = self.get_board_score(self.history[index][0], self.team)
            my_ending_points, enemy_ending_points = self.get_board_score(self.history[index+1][0], self.team)

            my_loss = my_starting_points - my_ending_points
            enemy_loss = enemy_starting_points - enemy_ending_points

            points = enemy_loss - my_loss
            p = min([1, points / 9])
            p = max([-1, p])

            return p

        return 0

    def finalize_data(self, reward: float, draw=False) -> list:
        labeled_data = []
        enemy_team = 1 if self.team != 1 else -1
        base_reward = 0

        if draw:
            my_score, enemy_score = self.get_board_score(self.history[-1][1], self.team)

            if my_score > enemy_score:
                labeled_data.append([flatten_board(self.history[-1][1]), reward])

        else:
            for i in range(len(self.history)-1):
                turn_reward = base_reward
                if self.boosted_rewards:
                    turn_reward += self.move_reward_generator(i)*0.33334

                labeled_data.append([flatten_board(self.history[i][1]), turn_reward])
            labeled_data.append([flatten_board(self.history[-1][1]), reward])

        self.distribute_rewards(labeled_data)

        return labeled_data

    def distribute_rewards(self, labeled_data):
        if len(labeled_data) > 0:
            distributed_rewards = [0]*len(labeled_data)
            for index, entry in enumerate(labeled_data):
                reward = entry[1]*1
                for i in range(index):
                    reward *= self.gamma
                    distributed_rewards[index-i] += reward

            highest = abs(max(distributed_rewards, key=abs))
            if highest != 0:
                for i in range(len(distributed_rewards)):
                    labeled_data[i][1] = distributed_rewards[i] / highest




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
        starting_board = self.encode_env(env, self.team)
        action = self.engine.play(env, chess.engine.Limit(time=self.time_limit)).move
        env.push(action)
        chosen_board = self.encode_env(env, self.team)
        self.history.append([starting_board, chosen_board, action])
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
            epsilon=0.98,
    ):
        super().__init__(team, encoder, score_encoder, boosted_rewards)
        self.model = model
        self.epsilon = epsilon
        self.max_varience = 0.05
        self.random_count = 0
        self.greedy = False

    def terminal_balancer(self, env, encoded_board, team, reverse=False) -> float:
        result = env.result()
        my_score, enemy_score = self.get_board_score(encoded_board, team, reverse=reverse)

        if result != "*":
            if result == "1-0":
                if team == 1:
                    return 1
                else:
                    return -1

            elif result == "0-1":
                if team == -1:
                    return 1
                else:
                    return -1

            elif result == "1/2-1/2":
                ratio = my_score / enemy_score
                return 0.5 if ratio <= 0.85 else -0.5


        else:
            legal_moves = list(env.legal_moves)
            ratio = enemy_score / my_score
            for action in legal_moves:
                env.push(action)
                result = env.result()
                env.pop()

                if result == "1-0" and team != 1:
                    return -1

                elif result == "0-1" and team != -1:
                    return -1

                elif result == "1/2-1/2":
                    if ratio < 1:
                        return -0.5

        return 0

    def get_enemy_move(self, env) -> float:
        starting_board = self.encode_env(env, self.team*-1)
        best_score = -inf
        for action in env.legal_moves:
            env.push(action)
            encoded_board = self.encode_env(env, self.team * -1)
            pred_board = flatten_board(encoded_board)
            pred_board = np.array([pred_board])
            pred = fast_predict(pred_board, self.model)
            #pred += self.terminal_balancer(env, encoded_board, self.team * -1, reverse=True)
            env.pop()
            if pred > best_score:
                best_score = pred

        return best_score

    def get_greedy_move(self, env):
        starting_board = self.encode_env(env, self.team)
        best_move = None
        chosen_board = None
        enemy_team = self.team * -1
        enemy_score = self.get_board_score(starting_board, self.team)[1]
        best_score = -inf
        difference = 0
        enemy_score = 0
        starting = str(env)

        for action in env.legal_moves:
            env.push(action)
            encoded_board = self.encode_env(env, self.team)

            resulting_score = self.get_board_score(encoded_board, self.team)[1]
            player_attack = enemy_score - resulting_score
            enemy_attack = self.get_enemy_greedy_move(env)[0]
            pred = player_attack - enemy_attack

            if pred > best_score or (pred == best_score and (abs(player_attack) + abs(enemy_attack)) > difference):
                best_score = pred
                best_move = action
                chosen_board = encoded_board
                difference = player_attack - enemy_attack
                enemy_score = enemy_attack
                ending = str(env)

            env.pop()

        return best_move, difference, chosen_board

    def generate_move(self, env):
        starting_board = self.encode_env(env, self.team)
        best_move = None
        chosen_board = None
        best_score = -inf
        legal_moves = list(env.legal_moves)

        if not self.greedy and random.random() >= self.epsilon:
            move = random.sample(list(env.legal_moves), 1)[0]
            env.push(move)
            encoded_board = self.encode_env(env, self.team)
            best_move = move
            chosen_board = encoded_board
            self.random_count += 1
            env.pop()

        if best_move is None:
            if self.greedy:
                greedy_move, diff, greedy_board = self.get_greedy_move(env)
            else:
                diff = 0

            if diff != 0:
                best_move = greedy_move
                chosen_board = greedy_board
            else:
                moves_list = []

                for action in legal_moves:
                    env.push(action)
                    encoded_board = self.encode_env(env, self.team)
                    pred_board = flatten_board(encoded_board)
                    pred_board = np.array([pred_board])
                    pred = fast_predict(pred_board, self.model)
                    pred += self.repeat_move_adjuster(action, -4)

                    if pred > best_score:
                        best_score = pred

                    moves_list.append([pred, action, encoded_board])
                    env.pop()

                moves_list.sort(key=lambda x: float(x[0]), reverse=True)
                highest_index = 0
                for m in moves_list:
                    highest_index += 1
                    if best_score - m[0] < self.max_varience:
                        break
                move = random.choice(moves_list[:highest_index])
                best_move = move[1]
                chosen_board = move[2]

        if best_move is None:
            best_move = legal_moves[0]

        if chosen_board is not None:
            self.history.append([starting_board, chosen_board, best_move])

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
        self.greedy = False

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
            width=100,
            speed_hacks=False,
    ):
        super().__init__(team, encoder, score_encoder, boosted_rewards, model, epsilon)
        self.depth = depth
        self.width = width
        self.speed_hacks = speed_hacks
        self.speed_barrier = 0


    def recursive_move_selector(self, env, team, legal_moves, depth):
        _env = env.copy(stack=False)
        starting_board = self.encode_env(_env, team)
        if depth <= 1:
            temp = self._generate_move(_env, team)
            return temp[1], temp[0]
        best_score = -999999
        best_move = None

        shallow_results = []

        for action in env.legal_moves:
            env.push(action)
            encoded_board = self.encode_env(env, team)
            pred_board = np.array([starting_board+encoded_board])
            pred = fast_predict(pred_board, self.model)
            #pred += self.terminal_balancer(env, encoded_board, team)
            pred += self.repeat_move_adjuster(action, -4)
            env.pop()
            shallow_results.append([pred, action])

        sorted_results = sorted(shallow_results, key=lambda x: x[0], reverse=True)

        projected_data = []

        for i in range(min(len(sorted_results), self.width)):
            move = sorted_results[i][1]
            _env.push(move)
            pred = sorted_results[i][0]
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

    def generate_move(self, env):
        starting_board = self.encode_env(env, self.team)
        legal_moves = list(env.legal_moves)
        if len(legal_moves) > 1:
            ideal_move = self.recursive_move_selector(
                env, self.team, legal_moves, self.depth
            )[1]
            if type(ideal_move) == str:
                ideal_move = random.sample(legal_moves, 1)[0]
                print(f"Had to pick random move! {len(legal_moves)}")

        else:
            ideal_move = legal_moves[0]
        env.push(ideal_move)
        chosen_board = self.encode_env(env, self.team)

        self.history.append([starting_board, starting_board+chosen_board, ideal_move])
        return ideal_move

    def _generate_move(self, env, team):
        starting_board = self.encode_env(env, team)
        best_move = None
        best_score = -inf

        for action in env.legal_moves:
            env.push(action)
            encoded_board = self.encode_env(env, team)
            pred_board = np.array([starting_board+encoded_board])
            pred = fast_predict(pred_board, self.model)
            #pred += self.terminal_balancer(env, encoded_board, team)
            env.pop()
            if pred > best_score:
                best_move = action
                best_score = pred
        return best_move, best_score

    def __repr__(self):
        return f"Distiller bot {'WHITE' if self.team == 1 else 'BLACK'} "


if __name__ == "__main__":
    x = [1,2,3]
    print(x[:0+1])
