import json
from rps_game_classes.player import Player
from utils.utils import *

# Defines which element beats another element in the Rock, Paper, Scissors game
rules = {
    'rock': 'scissors',
    'scissors': 'paper',
    'paper': 'rock'
}


# Function to determine the winner of a round of the game
def get_round_winner(player1_choice, player2_choice):
    if player1_choice == player2_choice:
        return 0  # Tie
    elif rules[player1_choice] == player2_choice:
        return 1  # Player 1 wins
    return 2  # Player 2 wins


# Function to add game results to the player
def add_game_result_to_player(player, is_win, players):
    if player not in players:
        players[player] = Player(player)  # Create a new Player object if not exists
    players[player].add_game(is_win)  # Add the game result to the player


# Function to parse a line of game results and update players' scores
def update_players_from_game_result(line, players):
    game_round = line.strip().split()
    player1, player1_choice, player2, player2_choice = game_round
    round_winner = get_round_winner(player1_choice, player2_choice)
    add_game_result_to_player(player1, round_winner == 1, players)
    add_game_result_to_player(player2, round_winner == 2, players)


# Function to play the game and determine the winner
def game(results_filename):
    print(f'Starting the game with {results_filename}')
    check_file_exists(results_filename)
    players = {}
    with open(results_filename, 'r', encoding='utf8') as file:
        file.readline()  # Skip header line
        for line in file:
            update_players_from_game_result(line, players)
    return determine_winner(players)


# Function to determine the winner based on player proportions
def determine_winner(players):
    leader_board = sorted(players.values(), key=lambda x: x.get_proportion(), reverse=True)
    if leader_board[0].get_proportion() == leader_board[1].get_proportion():
        return 'tie'
    else:
        return leader_board[0].get_name()


if __name__ == '__main__':
    # Load configuration from a JSON file
    with open('config-rps.json', 'r') as json_file:
        config = json.load(json_file)

if __name__ == '__main__':
    with open('config-rps.json', 'r') as json_file:
        config = json.load(json_file)

    winner = game(config['results_filename'])
    print(f'the winner is: {winner}')
