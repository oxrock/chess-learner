import gym
import gym_chess
from players import Player, ML_Player, ML_Player_Trainer
from chessbot_utils import (
    create_piece_dict,
    load_memorybank,
    load_model,
    save_data,
    load_data,
    create_training_data,
    train_model,
)
from multiprocessing import Process, Queue
import timeit
import time


class Trainer:
    def __init__(self, training_method: int):
        self.training_method = training_method  # 0: random mover opponent,1:random mirrors, 2:old version opponent, 3:mirror ML match
        self.env = gym.make("Chess-v0")
        self.piece_encoder = create_piece_dict()
        self.game_count = 1
        self.model, self.memorybank = load_data(
            "chess_model-v1.mdl", self.piece_encoder, self.env
        )
        self.old_model = load_model(
            "chess_model-trainer_1.mdl", self.piece_encoder, self.env
        )
        self.decisive_games = 0
        self.decisive_games_target = 5
        self.stat_tracker = {"white_wins": 0, "black_wins": 0, "draws": 0}

    def play_match(self, player_white, player_black) -> (Player, Player, str):
        state = self.env.reset()
        done = False
        turn = state.turn
        observation = state

        while not done:
            if turn:
                action = player_white.generate_move(observation.copy(stack=False))
            else:
                action = player_black.generate_move(observation.copy(stack=False))

            observation, reward, done, _ = self.env.step(action)
            turn = observation.turn

        return (player_white, player_black, observation.result())

    def generate_players(
        self,
    ) -> (
        Player,
        Player,
    ):  # handles alternating colors to keep training samples evenly distributed
        if self.training_method == 0:
            if self.game_count % 2 == 0:
                white_player = Player(1, self.piece_encoder, True)
                black_player = ML_Player(0, self.piece_encoder, True, self.model)
                print("ML is black")
            else:
                white_player = ML_Player(1, self.piece_encoder, True, self.model)
                black_player = Player(0, self.piece_encoder, True)
                print("ML is white")

        elif self.training_method == 1:
            white_player = Player(1, self.piece_encoder, True)
            black_player = Player(0, self.piece_encoder, True)

        elif self.training_method == 2:
            if self.game_count % 2 == 0:
                white_player = ML_Player(1, self.piece_encoder, True, self.model)
                black_player = ML_Player_Trainer(0, self.piece_encoder, True, self.old_model)
            else:
                white_player = ML_Player_Trainer(1, self.piece_encoder, True, self.old_model)
                black_player = ML_Player(0, self.piece_encoder, True, self.model)
        elif self.training_method == 3:
            white_player = ML_Player(1, self.piece_encoder, True, self.model)
            black_player = ML_Player(0, self.piece_encoder, True, self.model)

        else:
            print(f"training method '{self.training_method}' not recognized!")
            print("spawning random movers")
            white_player = Player(1, self.piece_encoder, True)
            black_player = Player(0, self.piece_encoder, True)

        return white_player, black_player

    def train(self):
        while True:
            white_player, black_player = self.generate_players()

            print(f"Starting game #{self.game_count}")
            _w, _b, result = self.play_match(white_player, black_player)

            if result == "1-0":
                print("Winner: White")
                self.stat_tracker["white_wins"] += 1
                white_data = white_player.finalize_data(1)
                black_data = black_player.finalize_data(-1)
            elif result == "0-1":
                print("Winner: Black")
                self.stat_tracker["black_wins"] += 1
                white_data = white_player.finalize_data(-1)
                black_data = black_player.finalize_data(1)
            else:
                print("Draw")
                self.stat_tracker["draws"] += 1
                white_data = white_player.finalize_data(0)
                black_data = black_player.finalize_data(0)

            if result in ["0-1", "1-0"]:
                for m in white_data + black_data:
                    self.memorybank.append(m)
                self.decisive_games += 1
            self.game_count += 1

            # if self.game_count % 5 == 0:
            if self.decisive_games == self.decisive_games_target:
                self.decisive_games_target += 5
                train_model(self.model, self.memorybank)
                save_data("chess_model-v1.mdl", self.model, self.memorybank)

            for stat in self.stat_tracker.items():
                print(stat)
            print("================\n")


def classless_trainer(
    q: Queue, training_method: int,rewards_boosting:bool
):  # 0: random mover opponent,1:random mirrors, 2:old version opponent, 3:mirror ML match
    env = gym.make("Chess-v0")
    piece_encoder = create_piece_dict()
    model = load_model("chess_model-v1.mdl", piece_encoder, env)
    old_model = load_model("chess_model-trainer_1.mdl", piece_encoder, env)

    def generate_players(game_count: int) -> (Player, Player):
        if training_method == 0:
            if game_count % 2 == 0:
                white_player = Player(1, piece_encoder, rewards_boosting)
                black_player = ML_Player(0, piece_encoder, rewards_boosting, model)
                # print("ML is black")
            else:
                white_player = ML_Player(1, piece_encoder, rewards_boosting, model)
                black_player = Player(0, piece_encoder, rewards_boosting)
                # print("ML is white")

        elif training_method == 1:
            white_player = Player(1, piece_encoder, rewards_boosting)
            black_player = Player(0, piece_encoder, rewards_boosting)

        elif training_method == 2:
            if game_count % 2 == 0:
                white_player = ML_Player(1, piece_encoder, rewards_boosting, model)
                black_player = ML_Player_Trainer(0, piece_encoder, rewards_boosting, old_model)
            else:
                white_player = ML_Player_Trainer(1, piece_encoder, rewards_boosting, old_model)
                black_player = ML_Player(0, piece_encoder, rewards_boosting, model)
        elif training_method == 3:
            white_player = ML_Player(1, piece_encoder, rewards_boosting, model)
            black_player = ML_Player(0, piece_encoder, rewards_boosting, model)

        else:
            # print(f"training method {self.training_method} not recognized!")
            # print("spawning random movers")
            white_player = Player(1, piece_encoder, rewards_boosting)
            black_player = Player(0, piece_encoder, rewards_boosting)

        return white_player, black_player

    def play_match(player_white, player_black) -> (Player, Player, str):
        state = env.reset()
        done = False
        turn = state.turn
        observation = state

        while not done:
            if turn:
                action = player_white.generate_move(observation.copy(stack=False))
            else:
                action = player_black.generate_move(observation.copy(stack=False))

            observation, reward, done, _ = env.step(action)
            turn = observation.turn

        return (player_white, player_black, observation.result())

    def train(updates=False):
        game_count = 0
        decisive_games = 0
        recorded_draws = 0
        update_count = 0
        while True:
            white_player, black_player = generate_players(game_count)
            _w, _b, result = play_match(white_player, black_player)

            if result == "1-0":
                print(f"{str(white_player)} victory over {str(black_player)}")
                white_data = white_player.finalize_data(1)
                black_data = black_player.finalize_data(-1)


            elif result == "0-1":
                print(f"{str(black_player)} victory over {str(white_player)}")
                white_data = white_player.finalize_data(-1)
                black_data = black_player.finalize_data(1)

            else:
                white_data = []
                black_data = []

            if training_method == 0:
                if type(white_player) == ML_Player:
                    training_data = white_data

                elif type(black_player) == ML_Player:
                    training_data = black_data

                else:
                    print(f"invalid player types in training method 0")

            elif training_method == 1:
                training_data = white_data + black_data

            elif training_method == 2:
                if type(white_player) == ML_Player:
                    training_data = white_data

                elif type(black_player) == ML_Player:
                    training_data = black_data

                else:
                    print(f"invalid player types in training method 2")

            elif training_method == 3:
                training_data = white_data + black_data

            else:
                print(f"Training method {training_method} is unknown")

            if len(training_data) > 0:
                q.put(training_data)
                decisive_games += 1
                update_count += 1


            # else:
            #     if recorded_draws < decisive_games/2:
            #         q.put(white_data + black_data)
            #         recorded_draws += 1
            #         update_count += 1

            if updates and update_count >= 10:
                return None

            game_count += 1
            # print(game_count)


    while True:
        train(updates=True)
        print("Updating to current models...")
        model = load_model("chess_model-v1.mdl", piece_encoder, env)
        old_model = load_model("chess_model-trainer_1.mdl", piece_encoder, env)


def data_gobbler(q: Queue):
    env = gym.make("Chess-v0")
    encoder = create_piece_dict()
    model, memory_bank = load_data("chess_model-v1.mdl", encoder, env)
    # train_model(model,memory_bank,batch_size=len(memory_bank),epochs=2)
    # save_data("chess_model-v1.mdl", model, memory_bank)
    train_count = 0
    game_count = 0
    while True:
        if not q.empty():
            data = q.get()
            game_count += 1
            train_count += 1
            for item in data:
                memory_bank.append(item)
            if train_count > 10:
                train_model(model, memory_bank)
                save_data("chess_model-v1.mdl", model, memory_bank)
                train_count = 0
                print(f"{game_count} matches trained so far!")

            if game_count %1000 == 0:
                save_data("chess_model-trainer_1.mdl", model, memory_bank)

if __name__ == "__main__":
    try:
        # trainer = Trainer(training_method=0)
        # trainer.train()
        my_q = Queue()
        fo_pool = [Process(target=classless_trainer, args=(my_q, 2,True)) for i in range(6)]
        for p in fo_pool:
            p.start()
        data_gobbler(my_q)

    except KeyboardInterrupt:
        for p in fo_pool:
            p.terminate()
        input("Enter to exit!")
