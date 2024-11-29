import random
from typing import List, Callable
import itertools
from tqdm import tqdm
import numpy as np

class Card:
    def __init__(self, rank: int, suit: str):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        rank_raster = self.rank

        if rank_raster == 10:
            rank_raster = "King"
        elif rank_raster == 9:
            rank_raster = "Queen"
        elif rank_raster == 8:
            rank_raster = "Jack"

        if self.suit == "bello":
            return f"{self.rank} {self.suit}"
        else:
            return f"{self.rank} di {self.suit}"
        
class Deck:
    suits = ['picche', 'bello', 'fiori', 'cuori']
    ranks = list(range(1, 11))  # Ranks from 1 to 7, plus 8, 9, and 10 for face cards.

    def __init__(self):
        self.cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_cards: int) -> List[Card]:
        return [self.cards.pop() for _ in range(num_cards)]
    
    def __str__(self):
        result = '#' * 10 + f' Deck {self.__hash__()} ' + '#' * 10 + '\n'
        for card in self.cards:
            result += str(card) + '\n'
        result += '#' * 20 + '\n'
        result += f'{len(self.cards)} cards in the deck.\n'
        for suit in self.suits:
            result += f'{sum(1 for card in self.cards if card.suit == suit)} {suit}\n'
        result += '#' * 20
        return result

    def reset(self):
        self.cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]
        self.shuffle()

class Player:
    def __init__(self, side: int):
        if side not in [1, 2]:
            raise ValueError("Side must be 1 or 2.")
        self.side = side
        self.hand = []
        self.captures = []
        self.scopas = 0


    def play_card(self, card_index: int, v= 0) -> Card:
        card = self.hand.pop(card_index)
        if v >= 2: print(f'[PLAYER] Player {self.__hash__()} played {card}.')
        return card

    def capture(self, cards: List[Card], _with: Card):
        for card in cards:
            self.captures.append(card)
        if _with is not None and _with in self.hand: self.hand.remove(_with)

    def scopa(self):
        self.scopas += 1
    

    def __str__(self):
        return f'[PLAYER] Player {self.__hash__()} for side {self.side} has {len(self.hand)} cards in hand and {len(self.captures)} captures.'
    
    def show_hand(self):
        out = '#' * 10 + f' Player {self.__hash__()} ' + '#' * 10 + '\n'
        for card in self.hand:
            out += str(card) + '\n'
        out += '#' * 20
        return out
    
    def reset(self):
        self.hand = []
        self.captures = []
        self.scopas = 0

class ScoponeGame:
    def __init__(self):
        self.deck = Deck()
        self.players = [Player(i) for i in [1,2,1,2]]
        self.table = []
        self.last_capture = self.players[0]
        self.step_points = [0, 0]
        self.match_points = [0, 0]
        self.game_tick = 0
        self.match_tick = 0
        
    def deal_initial_hands(self):
        self.deck.reset()
        for player in self.players:
            player.reset()
            player.hand = self.deck.deal(10)
        self.gt()
        self.mt()

        
    def gt(self):
        self.game_tick += 1
    
    def mt(self):
        self.match_tick += 1

    def __str__(self):
        return f"Players: {[player.__hash__() for player in self.players]}, Table: {self.table}"
    
    def player_details(self):
        return [str(player) for player in self.players]
    
    def card_in_table(self, card):

        if len(self.table) == 0:
            return False, []
        

        current_table = [self.table[i] for i in range(len(self.table))]
        all_combinations = []
        for i in range(1, len(current_table) + 1):
            all_combinations.extend(list(itertools.combinations(current_table, i)))

        for comb in all_combinations:
            for c in comb:
                if isinstance(c, list):
                    if sum([cc.rank for cc in c]) == card.rank:
                        return True, [c]
                    
            if c.rank == card.rank:
                return True, [c]
        return False, []

    def __describe_status(self) -> str:
        out = '#' * 10 + ' Game Status ' + '#' * 10 + '\n'
        out += 'Table:\n'
        for card in self.table:
            out += str(card) + '\n'
        out += '#' * 20 + '\n'
        for player in self.players:
            out += 'Player ' + str(player.__hash__()) + f' for side {player.side}\n'
            out += 'Hand:\n'
            out += player.show_hand() + '\n'
            out += 'Captured stack:\n'
            out += f'{[str(c) for c in player.captures]}\n'
        out += '#' * 20 + '\n'

        return out
        
    
    def play_card(self, card, player, v=0):
        if v >= 2: print(f'[GAME] Player {player.__hash__()} plays {card}')

        # ACE CASE
        if card.rank == 1:
                self.table.append(card)
                player.capture(self.table, _with=card)
                if v >= 2: 
                    print(f'[GAME] Player {player.__hash__()} captures {[str(c) for c in self.table]} with {card}')
                self.table=[]
                return


        isin, comb = self.card_in_table(card=card)

        if isin:
            self.last_capture = player
            # TODO comb is important as the agent will have to be able to chose the best capture
            

            for c in comb:
                self.table.remove(c)

            comb.append(card)
            player.capture(comb, _with=card)

            if self.table == []:
                player.scopa()
                if v >= 2: print(f'[GAME] Player {player.__hash__()} scopa!')
            
            if v >= 2: 
                print(f'[GAME] Player {player.__hash__()} captures {[str(c) for c in comb]} with {card}')
        else:
            if card in player.hand:
                player.play_card(player.hand.index(card), v=v)
            else:
                raise ValueError(f'Card {card} not found in player\'s hand.')
            self.table.append(card)
        self.gt()
        self.mt()

    def evaluate_round(self, players: List[Player], v=0) -> List[int]:

        side1_points = 0
        side2_points = 0

        

        side1, side2 = self.balance_captures()
        
        for player in players:
            if player.scopas > 0:
                if player.side == 1:
                    side1_points += player.scopas
                elif player.side == 2:
                    side2_points += player.scopas

        if len(side1) + len(side2) != 40:
            raise ValueError(f"Not all cards have been captured. Side 1 has {len(side1)} and Side 2 has {len(side2)}")

        # Key evaulation

        if v >= 2: print(f'[EVAL] Side 1: {side1_points} Side 2: {side2_points}. Next up: Sette Bello')
        
        #SetteBello
        for card in side1:
            if card.rank == 7 and card.suit == 'bello':
                side1_points += 1
                break
        if side1_points == 0:
            side2_points += 1


        if v >= 2: print(f'[EVAL] Side 1: {side1_points} Side 2: {side2_points}. Next up: Cards')
        #Cards
        # Only possible tie is 20 cards each and in that case no points are awarded
        if len(side1) > len(side2):
            side1_points += 1
        elif len(side1) < len(side2):
            side2_points += 1


        if v >= 2: print(f'[EVAL] Side 1: {side1_points} Side 2: {side2_points}. Next up: Ori')
        #Ori
        counter = 0
        for card in side1:
            if card.suit == 'bello':
                counter += 1
        if counter > 5:
            side1_points += 1
        elif counter < 5:
            side2_points += 1

        if v >= 2: print(f'[EVAL] Side 1: {side1_points} Side 2: {side2_points}. Next up: Primiera')
        #primiera
        score1 = [0,0,0,0]
        score2= [0,0,0,0]
        for i,suit in enumerate(['bello', 'picche', 'fiori', 'cuori']):
            for card in side1:
                if card.suit == suit and card.rank >= score1[i]:
                    score1[i] = card.rank
            for card in side2:
                if card.suit == suit and card.rank >= score2[i]:
                    score2[i] = card.rank
        if sum(score1) > sum(score2):
            side1_points += 1
        elif sum(score1) < sum(score2):
            side2_points += 1

        if v >= 2: print(f'[EVAL] Side 1: {side1_points} Side 2: {side2_points}. Next up: Napola')

        #Napola

        side1_belli = []
        side2_belli = []

        for card in side1:
            if card.suit == 'bello':
                side1_belli.append(card)
        for card in side2: 
            if card.suit == 'bello':
                side2_belli.append(card)

        
        side1_belli_ranks = sorted([card.rank for card in side1_belli])
        side2_belli_ranks = sorted([card.rank for card in side2_belli])

        def calculate_sequence_points(ranks):
            points = 0
            if all(rank in ranks for rank in [1, 2, 3]):
                points = 3
                for rank in range(4, 11):
                    if rank in ranks:
                        points += 1 
                    else:
                        break
            return points

        side1_points += calculate_sequence_points(side1_belli_ranks)
        side2_points += calculate_sequence_points(side2_belli_ranks)

        if v >= 2: print(f'[EVAL] Final Score - Side 1: {side1_points} Side 2: {side2_points}')

        return [side1_points, side2_points]

    def balance_captures(self):
        side1_captures = []
        side2_captures = []
        for player in self.players:
            if player.side == 1:
                side1_captures += player.captures
            elif player.side == 2:
                side2_captures  += player.captures 
        
        return side1_captures, side2_captures

        
    
    
    
    def __play_game(self, v = 0):
        i=0
        while [len(player.hand) == 0 for player in self.players] != [True, True, True, True]:
            if v >= 2: print('#' * 20 + f' Turn {i+1} ' + '#' * 20)
            for player in self.players:
                self.play_card(player.hand[random.randint(0, len(player.hand) - 1)], player, v=v)
            if v >= 2: print('#'*48)

            if [len(player.hand) == 0 for player in self.players] == [True, True, True, True]:
                self.last_capture.capture(self.table, _with=None)
                if v >= 2: print(f'[GAME] Player {self.last_capture.__hash__()} captures the table.')
                if v >= 2: print(f'[GAME] {[str(c) for c in self.table]}')
                self.table = []
                if v >= 2: print('[GAME] \n\n\n>>>>>>>>>>>>>>>>Game over!\n\n\n')
                return
            i+=1

    def is_match_over(self, side1_score, side2_score, winning_threshold = 21, v = 0):

        if abs(side1_score-side2_score) == 1 and min(side1_score, side2_score) >= winning_threshold-1:
            if v >= 1: print(f'[MATCH] DEUCE! old threshold: {winning_threshold} new threshold: {max(side1_score, side2_score) + 1}')
            winning_threshold = max(side1_score, side2_score) + 1

        if side1_score >= winning_threshold and side2_score < side1_score - 1:
            if v >= 0: print(f'[MATCH] Side 1 wins with {side1_score} points!')
            return True
        elif side2_score >= winning_threshold and side1_score < side2_score - 1:
            if v >= 0: print(f'[MATCH] Side 2 wins with {side2_score} points!')
            return True

        return False

    def __play_match(self, v = 0, winning_threshold = 21):
        if v == 1: print(f'[MATCH] Starting match with winning threshold {winning_threshold}')
        side1_score = []
        side2_score = []
        i = 0
        while not self.is_match_over(sum(side1_score), sum(side2_score), winning_threshold, v=v):
            self.deal_initial_hands()
            self.play_game(v=v)
            if v>= 2: print(self.describe_status())
            scores = self.evaluate_round(self.players, v=v)
            side1_score.append(scores[0])
            side2_score.append(scores[1])
            if v >= 1: print(f'[MATCH] ROUND {i+1} \t|\tSide 1:\t{sum(side1_score)} Side 2:\t{sum(side2_score)}')
            i+=1

            # Shift player to the right this imitates the rotation in the real game
            upper = self.players[1:]
            lower = self.players[:1]
            self.players = upper + lower

            if i > 50:
                raise ValueError("Too many rounds played.")
        if v >= 1: print(f'[MATCH] --------\n[MATCH] RESULTS \t|\tSide 1:\t{sum(side1_score)} Side 2:\t{sum(side2_score)} ')

        
        if sum(side1_score) > sum(side2_score):
            return 1
        else:
            return 2
        
    def reset(self, soft=False):
        self.deck.reset()
        #self.players = [Player(i) for i in [1,2,1,2]]
        self.table = []
        self.last_capture = None
        self.step_points = [0, 0]
        if not soft: self.match_points = [0, 0]
        self.game_tick = 0
        self.match_tick = 0


    
    def initialise_actions(self, player: Player, v = 0):

        actions_array = [0] * 40

        for card in player.hand:

            indx = card.rank + 30 * (card.suit == 'bello') + 20 * (card.suit == 'fiori') + 10 * (card.suit == 'picche') - 1

            if v>=2: print(f'Card {card} has index {indx}')
            
            actions_array[indx] = 1

            isin, comb = self.card_in_table(card=card)
            if isin:
                #actions_array[indx] = {'type': 'capture', 'card': str(card), 'with': [str(c) for c in comb], 'leaving': len(self.table)-len(comb)}
                actions_array[indx] = 2


        if v >= 2: print(f'[DEBUG] Player {player.__hash__()} actions: {actions_array}')
        return actions_array
        return {
            i: {'type': 'play', 'card': str(player.hand[i])} for i in range(len(player.hand))
        }
    

    def get_player_actions(self, player: Player, v=0):

        
        return self.initialise_actions(player,v=v)

        actions = self.initialise_actions(player)
        
        for card in player.hand:
            isin, comb = self.card_in_table(card=card)
            if isin:
                actions[player.hand.index(card)] = {'type': 'capture', 'with': str(card), 'card': [str(c) for c in comb], 'leaving': len(self.table)-len(comb)}

        return self.initialise_actions(player)
    

    def map_rank(self, c: Card):
        if c.rank == 1:
            return 12
        elif c.rank == 7:
            return 11
        else:
            return c.rank
        
    def map_card_index(self, card: Card):
        return card.rank + 30 * (card.suit == 'bello') + 20 * (card.suit == 'fiori') + 10 * (card.suit == 'picche') - 1
        

    def get_player_state(self, player: Player, v = 0):
        hand = [(card.rank, card.suit) for card in player.hand]
        current_table = [(card.rank, card.suit) for card in self.table]
        s1, s2 = self.balance_captures()
        if player.side == 1:
            captures = s1
        else:
            captures = s2

        state_indexes = {'card':[self.map_card_index(card) for card in player.hand],'table':[self.map_card_index(card) for card in self.table],'capt':[self.map_card_index(card) for card in captures]}

        

        state = np.zeros((3,40))

        j=0
        for i in ['card', 'table', 'capt']:
            for index in state_indexes[i]:

                state[j][index] = 1
            j+=1

        return state


    

    def __oldget_player_state(self, player: Player, v = 0):
        hand = [(card.rank, card.suit) for card in player.hand]
        current_table = [(card.rank, card.suit) for card in self.table]

        captures = []
        for p in self.players:
            if p.side == player.side:
                captures += [(card.rank, card.suit) for card in p.captures]

        {
            'hand': hand,
            'captures': captures,
            'table': current_table,
        }
        
        if player.side == 1:
            deltapoints = self.step_points[1] - self.step_points[0]
        else:
            deltapoints = self.step_points[0] - self.step_points[1]
       
        cardsplayed = 10 - len(player.hand)

        cardsontable = len(self.table)

        hand_eval = []

        for card in player.hand:
            isin, comb = self.card_in_table(card=card)
            if isin:
                w1 = 3
            else:  
                w1 = 1
            rank_importance = (w1 * self.map_rank(card) / 36 ) * (1.0 - (0.01 * self.game_tick))

            if card.suit == 'bello':
                w2 = 1
            else:
                w2 = 1/3

            suit_importance = w2

            hand_eval.append(rank_importance * suit_importance)

        
        hand_eval.extend([0] * (10 - len(hand_eval)))
        
        table_eval = []

        for card in self.table:
            rank_importance = (self.map_rank(card) / 12 ) * (1.0 - (0.01 * self.game_tick))

            if card.suit == 'bello':
                w2 = 1
            else:
                w2 = 1/3

            suit_importance = w2

            table_eval.append(rank_importance * suit_importance)

        table_eval.extend([0] * (10 - len(table_eval)))
    
        

        if v >= 1: print(f'[STATE] Player {player.__hash__()} state: \n {[deltapoints, cardsplayed, cardsontable, self.game_tick, self.match_tick] + hand_eval + table_eval  + [7] + [len(r.hand) for r in self.players] } for length {len([deltapoints, cardsplayed, cardsontable, self.game_tick, self.match_tick] + hand_eval + table_eval)}')


        return [self.game_tick, self.match_tick,self.step_points[0], self.step_points[1] , deltapoints, cardsplayed, cardsontable] + hand_eval + table_eval
    
    def calculate_reward(self, player: Player, card: Card,v=0):
        isin, comb = self.card_in_table(card=card)

        if comb is None:
            raise ValueError('Combination is None. Comb: ' + str(comb))
        

        reward = 0
    
        if isin:
            #scopa
            if len(self.table) - len(comb) == 0:
                reward += 10
            elif len(self.table) - len(comb) == 1:
                reward -= 5
            #settebello
            
            comb.append(card)

            for c in comb:
                if c.rank == 7 and c.suit == 'bello':
                    reward += 10
            #cards, ori and napola
            for c in comb:
                if c.suit == 'bello':
                    reward += 1 + c.rank*0.5
                else:
                    reward += 0.25 + c.rank*0.1
        

        return reward   
    
    def get_action(self, player: Player, action, v=0):
        
        for i, card in zip([self.map_card_index(card) for card in player.hand], player.hand):
            if i == action:
                return card 


    def gym_step(self, player: Player, action, v=-1):
        card = self.get_action(player, action, v=v)
        if card is None:
            raise ValueError('Card is None. Original action: ' + str(action))
        reward = self.calculate_reward(player, card, v=v)

        if v == -1: print(f'[RL] Action: {action} by player {player.__hash__()} yields reward {reward}')

        self.play_card(card, player, v=v)

        return self.get_player_state(player, v=v), reward, False, {}

    def random_step(self, player: Player, action, v=-1):
        possible = self.get_player_actions(player, v=v)
        chosen_index = np.argmax(possible)
        for i in range(4):
            rand = random.choice(possible)
            if rand == 1 or rand == 2:
                chosen_index = possible.index(rand)
                break
        
        card = self.get_action(player, chosen_index, v=v)
    
        self.play_card(card, player, v=v)

        return self.get_player_state(player, v=v)


    
    def __game_step(self, player: Player, action, v= 0):
        if self.is_match_over(self.step_points[0], self.step_points[1], 21):
            raise ValueError('Match is over. No more steps allowed.')
        card = action['card']

        if card is None:
            raise ValueError('No card selected. Original action: ' + str(action))
        if v >= 1: print(f'[GAME] Player {player.__hash__()} plays {card}')
        comp = card.split(' ')
        rank = int(comp[0])
        if len(comp) == 2:
            suit = comp[1]
        else:
            suit = comp[2]

        t = True
        for c in player.hand:
            if c.rank == rank and c.suit == suit:
                card = c
                t = False
                break

        if card not in player.hand or t:
            raise ValueError(f'Card {card} not in player\'s hand.')



        reward = self.calculate_reward(player, card)

        self.play_card(card, player, v=v)

        if sum([len(p.hand) == 0 for p in self.players]) == 4:
            self.last_capture.capture(self.table, _with=None)
            self.table = []
            eval = self.evaluate_round(self.players, v=v)
            self.match_points[0] += eval[0]
            self.match_points[1] += eval[1]
            if v == 1: print(f'[GAME] Game is over! {self.step_points[0]}|{self.step_points[1]} and {self.match_points[0]}|{self.match_points[1]}')

            for p in self.players:
                p.reset()
            self.deal_initial_hands()
            self.game_tick = 0
            
        return reward, self.is_match_over(self.match_points[0], self.match_points[1], 21)

        

        