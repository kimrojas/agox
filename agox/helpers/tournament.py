class Player:
    def __init__(self,tournament,generation,left=None):
        self.tournament = tournament

        if left is not None or generation <= 0:
            self.left = left
        else:
            self.left = Player(self.tournament,generation - 1)

        if generation > 0:
            self.right = Player(self.tournament,generation - 1)

        self.generation = generation
        self.counter = len(self.tournament.players)
        print('EXPANDING TREE: ',self)
        self.tournament.players.append(self)

    def __str__(self):
        t = f'{self.counter} {self.generation}'
        if self.generation > 0:
            t += f' {self.left.counter} {self.right.counter}'
        return t

class Tournament:
    def __init__(self):
        self.players = []

    def generation_of_player_with_largest_player_number(self):
        if len(self.players) == 0:
            return -1
        return self.players[-1].generation

    def get_player(self,number=0):
        while True:
            #print(len(self.players))                                                                                   
            if number < len(self.players):
                return self.players[number]
            # expand with a generation                                                                                  
            Player(self,self.generation_of_player_with_largest_player_number() + 1,
                   left=None if len(self.players) == 0 else self.players[-1])
