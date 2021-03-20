import gym
import gym_chess
from players import ML_Player,Player
from chessbot_utils import create_piece_dict,load_memorybank,load_model,save_data,load_data,create_training_data,train_model

class Trainer():
    def __init__(self,training_method:int):
        self.training_method = training_method #0: random mover opponent,1:random mirrors, 2:old version opponent, 3:mirror ML match
        self.env = gym.make("Chess-v0")
        self.piece_encoder = create_piece_dict()
        self.game_count = 1
        self.model, self.memorybank = load_data("chess_model-v1.mdl",self.piece_encoder, self.env)
        self.old_model = load_model("chess_model-trainer_1.mdl",self.piece_encoder,self.env)
        self.decisive_games = 0
        self.decisive_games_target = 5
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

    def generate_players(self)->(Player,Player): #handles alternating colors to keep training samples evenly distributed
        if self.training_method == 0:
            if self.game_count %2 == 0:
                white_player = Player(1,self.piece_encoder,True)
                black_player = ML_Player(0,self.piece_encoder,True,self.model)
                print("ML is black")
            else:
                white_player = ML_Player(1, self.piece_encoder,True,self.model)
                black_player = Player(0, self.piece_encoder,True)
                print("ML is white")

        elif self.training_method == 1:
            white_player = Player(1,self.piece_encoder,True)
            black_player = Player(0, self.piece_encoder,True)

        elif self.training_method == 2:
            if self.game_count % 2 == 0:
                white_player = ML_Player(1, self.piece_encoder,True, self.model)
                black_player = ML_Player(0, self.piece_encoder,True, self.old_model)
            else:
                white_player = ML_Player(1, self.piece_encoder, True,self.old_model)
                black_player = ML_Player(0, self.piece_encoder,True, self.model)
        elif self.training_method == 3:
            white_player = ML_Player(1, self.piece_encoder,True, self.model)
            black_player = ML_Player(0, self.piece_encoder,True, self.model)

        else:
            print(f"training method {self.training_method} not recognized!")
            print("spawning random movers")
            white_player = Player(1, self.piece_encoder,True)
            black_player = Player(0, self.piece_encoder,True)

        return white_player,black_player

    def train(self):
        while True:
            white_player,black_player = self.generate_players()

            print(f"Starting game #{self.game_count}")
            _w,_b,result = self.play_match(white_player,black_player)

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
                # white_data = white_player.finalize_data(-0.1)
                # black_data = black_player.finalize_data(-0.1)


            if result in ["0-1","1-0"]:
                for m in white_data + black_data:
                    self.memorybank.append(m)
                self.decisive_games+=1
            self.game_count += 1

            #if self.game_count % 5 == 0:
            if self.decisive_games == self.decisive_games_target:
                self.decisive_games_target+=5
                train_model(self.model, self.memorybank)
                save_data("chess_model-v1.mdl",self.model, self.memorybank)


            for stat in self.stat_tracker.items():
                print(stat)
            print('================\n')




if __name__ == "__main__":
    trainer = Trainer(training_method=1)
    trainer.train()
