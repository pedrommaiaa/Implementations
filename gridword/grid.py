import numpy as np

GRID_ROWS = 5
GRID_COLS = 5
START = (0, 0)
WIN_STATE = (4, 4)
 

class gridWorld:    

    def __init__(self, state=START):
        self.grid = np.zeros([GRID_ROWS, GRID_COLS])
        self.nStates = GRID_ROWS * GRID_COLS
        
        self.position = state
        self.done = False

        self.reward = 0

        # [left, right, up, down]    
        self.actions = [0, 1, 2, 3]
        self.nActions = len(self.actions)
         

    # take action
    def step(self, action):
        
        if action not in self.actions:
            raise ValueError("This is not a valid action.")

        # Left
        if action == 0:
            nextPos = (self.position[0], self.position[1]-1)

        # Right
        if action == 1:
            nextPos = (self.position[0], self.position[1]+1)
        
        # Up
        if action == 2:
            nextPos = (self.position[0]-1, self.position[1])
        
        # Down
        if action == 3:
            nextPos = (self.position[0]+1, self.position[1])


        if (nextPos[0] >= 0) and (nextPos[0] <= (GRID_ROWS - 1)):
            if (nextPos[1] >= 0) and (nextPos[1] <= (GRID_COLS - 1)):
                if nextPos != WIN_STATE:
                    self.position = nextPos


        if self.position == WIN_STATE:
            self.done = True

        return self.position, self.reward, self.done


    def reset(self):
        self.position = (0, 0)
        self.done = False


    def randomAction(self):
        return np.random.choice(self.actions)


    def display(self):
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                if (i, j) == self.position:
                    print('X', end=' ')
                elif (i, j) == (GRID_ROWS-1, GRID_COLS-1):
                    print(1, end=' ')
                else:
                    print(0, end=' ')
            print()

