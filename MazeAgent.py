import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import matplotlib.patches as patches


class MazeAgent:


    def __init__(self, **kwargs):

        self.agent_pos = np.array([0,0])
        self.target_pos = np.array([5,3])

        '''self.maze_dims = (8,4)
        self.maze_array = np.zeros(self.maze_dims)
        self.maze_array[3:5,1:3] = 1
        self.target_pos = np.array([5,3])'''

        self.maze_dims = (20,8)
        self.maze_array = np.zeros(self.maze_dims)
        self.maze_array[3:5,1:3] = 1
        self.maze_array[6,0:6] = 1
        self.maze_array[6:12,7] = 1
        self.maze_array[8:16,3] = 1
        self.maze_array[17,2:8] = 1
        self.target_pos = np.array([19,6])



        self.N_states = self.maze_dims[0]*self.maze_dims[1]
        #U, D, L, R
        self.N_actions = 4

        self.agent_circ_rad = 0.5
        self.target_circ_rad = 0.5*self.agent_circ_rad
        #self.resetTarget()

        #self.resetStateValues()
        #self.N_state_terms = len(self.getStateVec())



    def getStateVec(self):
        return(self.posToStateVec(self.agent_pos))


    def validMoves(self):
        #Gives you a list of the valid moves in a position,
        #where [0,1,2,3] is [u,d,l,r]
        moves = []
        x = self.agent_pos[0]
        y = self.agent_pos[1]

        if x>0 and self.maze_array[x-1, y] != 1:
            moves.append(2)
        if x<(self.maze_dims[0] - 1) and self.maze_array[x+1, y] != 1:
            moves.append(3)

        if y>0 and self.maze_array[x, y-1] != 1:
            moves.append(1)
        if y<(self.maze_dims[1] - 1) and self.maze_array[x, y+1] != 1:
            moves.append(0)

        return(moves)


    def stateVecToPos(self, sv):
        y, x = divmod(pos, self.maze_dims[0])
        return(np.array([x,y]))


    def posToStateVec(self, pos):
        #So if the pos is x, y, and the dims are Nx, Ny, we'll return
        #x + N_x*y.
        return(pos[0] + self.maze_dims[0]*pos[1])

    def iterate(self, action):

        #print('\niterating, action {} in pos {}'.format(action,self.agent_pos))
        #print('valid moves:',self.validMoves())
        if action in self.validMoves():
            if action==0:
                self.agent_pos += [0, 1]
            if action==1:
                self.agent_pos += [0, -1]
            if action==2:
                self.agent_pos += [-1, 0]
            if action==3:
                self.agent_pos += [1, 0]
        else:
            self.agent_pos = self.agent_pos

        #print('new pos after action:',self.agent_pos)
        return(self.reward(),self.getStateVec())


    def reward(self):
        if self.done():
            return(1)
        else:
            return(-0.01)


    def done(self):
        return(np.array_equal(self.agent_pos, self.target_pos))


    def addToHist(self):
        pass
        '''self.pos_hist = np.concatenate((self.pos_hist,[self.pos]))
        self.v_hist = np.concatenate((self.v_hist,[self.v]))
        self.t.append(self.t[-1] + self.time_step)
        self.r_hist.append(self.reward())'''



    def drawState(self,ax):

        ax.clear()
        ax.set_xlim((0,self.maze_dims[0]))
        ax.set_ylim((0,self.maze_dims[1]))

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        ax.set_xticks([])
        ax.set_yticks([])

        for x, y in np.ndindex(self.maze_dims):
            if self.maze_array[x,y] == 1:
                rect = patches.Rectangle((x,y),1,1,linewidth=1,edgecolor='Black',facecolor='Black')
                ax.add_patch(rect)


        agent_draw_pos = self.agent_pos + [0.5,0.5]
        agent_circ = plt.Circle(tuple(agent_draw_pos), self.agent_circ_rad, color='seagreen')
        ax.add_artist(agent_circ)

        target_draw_pos = self.target_pos + [0.5,0.5]
        target_circ = plt.Circle(tuple(target_draw_pos), self.target_circ_rad, color='tomato')
        ax.add_artist(target_circ)



    def plotStateParams(self,axes):

        '''ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]
        ax4 = axes[3]

        ax1.clear()
        ax1.plot(self.pos_hist[:,0][-1000:],label='x')
        ax1.plot(self.pos_hist[:,1][-1000:],label='y')
        ax1.legend()

        ax2.clear()
        ax2.plot(self.a_hist[-1000:],label='a')
        ax2.set_yticks([0,1,2,3])
        ax2.set_yticklabels(['U','D','L','R'])
        ax2.legend()


        ax3.clear()
        ax3.plot(self.r_hist[-1000:],label='R')
        ax3.legend()


        ax4.clear()
        ax4.plot(self.v_hist[:,0][-1000:],label='vx')
        ax4.plot(self.v_hist[:,1][-1000:],label='vy')
        ax4.legend()'''


    def initEpisode(self):
        self.agent_pos = np.array([0,0])
        #self.resetTarget()


































#
