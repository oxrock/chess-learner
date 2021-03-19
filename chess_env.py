import gym
import gym_chess
from players import ML_Player,Player
from chessbot_utils import create_piece_dict,load_memorybank,load_model,save_data,load_data,create_training_data,train_model

class Trainer():
    def __init__(self):
        self.env = gym.make("Chess-v0")
        self.piece_encoder = create_piece_dict()
        self.game_count = 0
        self.model, self.memorybank = load_data(self.piece_encoder, self.env)
        self.prediction_model = load_model(self.piece_encoder, self.env)
        self.stat_tracker = {"white_wins":0,
                             "black_wins":0,
                             "draws":0}

    def play_match(self,player_white,player_black)->(Player,Player,str):
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

        return (player_white,player_black,observation.result())

    def train(self):
        while True:
            white_player = ML_Player(1, self.piece_encoder, self.prediction_model, epsilon=0.75)
            black_player = ML_Player(0, self.piece_encoder, self.prediction_model, epsilon=0.75)

            print(f"Starting game #{self.game_count+1}")
            white_player,black_player,result = self.play_match(white_player,black_player)

            if result == "1-0":
                print('Winner: White')
                self.stat_tracker["white_wins"] += 1
                white_data = white_player.finalize_data(1)
                black_data = black_player.finalize_data(-1)
            elif result == "0-1":
                print('Winner: Black')
                self.stat_tracker["black_wins"] += 1
                white_data = white_player.finalize_data(-1)
                black_data = black_player.finalize_data(1)
            else:
                print("Draw")
                self.stat_tracker["draws"] += 1
                white_data = white_player.finalize_data(-0.1)
                black_data = black_player.finalize_data(-0.1)



            for m in white_data + black_data:
                self.memorybank.append(m)
            self.game_count += 1

            if self.game_count % 2 == 0:
                train_model(self.model, self.memorybank)
                save_data(self.model, self.memorybank)

            if self.game_count % 10 == 0:
                prediction_model = load_model(self.piece_encoder, self.env)
                print("Prediction model updated")

            for stat in self.stat_tracker.items():
                print(stat)
            print('================\n')




if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
