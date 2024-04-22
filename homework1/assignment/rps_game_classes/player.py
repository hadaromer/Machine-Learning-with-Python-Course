class Player:

    def __init__(self, name):
        self._name = name
        self._games = 0
        self._wins = 0

    def add_game(self,is_win):
        self._games += 1
        if is_win:
            self._wins +=1

    def get_proportion(self):
        return self._wins / self._games

    def get_name(self):
        return self._name