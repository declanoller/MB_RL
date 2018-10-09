from MazeAgent import MazeAgent
import matplotlib.pyplot as plt
from random import randint,random,sample
import numpy as np
from math import atan,sin,cos,sqrt,ceil,floor,log
from datetime import datetime
import FileSystemTools as fst
from collections import namedtuple
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.optim as optim

Experience = namedtuple('exp_tup',('s','a','r','s_next'))



class MBAgent:



    def __init__(self, **kwargs):

        self.agent_class = kwargs.get('agent_class',MazeAgent)
        self.agent_class_name = self.agent_class.__name__
        self.agent = self.agent_class(**kwargs)
        self.N_states = self.agent.N_states
        self.N_actions = self.agent.N_actions

        self.params = {}
        self.params['gamma'] = 1.0
        self.params['init_epsilon'] = 0.8
        self.params['epsilon'] = self.params['init_epsilon']
        self.params['epsilon_decay'] = .995


        #For the table based method, we need a P_s,a,s' matrix and an R_s,a matrix
        self.P = np.zeros((self.N_states,self.N_actions,self.N_states))
        self.N = np.zeros((self.N_states,self.N_actions))
        self.R = np.zeros((self.N_states,self.N_actions))

        self.dtype = torch.float32
        self.Q = torch.zeros((self.N_states,self.N_actions), requires_grad=True, dtype=self.dtype)

        self.optimizer = optim.RMSprop([self.Q])
        self.experiences = []
        self.R_tot_hist = [0]

        self.dir = kwargs.get('dir','misc_runs')
        self.date_time = kwargs.get('date_time',fst.getDateString())
        self.base_fname = 'test' + '_' + self.date_time
        self.img_fname = fst.combineDirAndFile(self.dir, self.base_fname + '.png')


        self.createFigure()
        self.plotAll()
        #self.showFig(block=False)





    def updateModel(self, xp):
        #Add experience xp to P, R, N
        N = self.N[xp.s, xp.a]
        self.P[xp.s, xp.a, xp.s_next] = (N/(N+1))*self.P[xp.s, xp.a, xp.s_next] + 1.0/(N+1)
        self.R[xp.s, xp.a] = (N/(N+1))*self.R[xp.s, xp.a] + xp.r/(N+1)
        self.N[xp.s, xp.a] += 1.0

        self.experiences.append(xp)

    def updateEpsilon(self):
        self.params['epsilon'] *= self.params['epsilon_decay']

    def modelSimulatedExperience(self, s, a):

        s_next = np.random.choice(list(range(len(self.P[s, a]))),p=self.P[s, a])
        r = torch.tensor(self.R[s, a])
        return(r,s_next)


    def greedyAction(self,state_ind):
        return(torch.argmax(self.Q[state_ind]).detach())


    def epsGreedyAction(self,state_ind):
        if random()>self.params['epsilon']:
            return(self.greedyAction(state_ind))
        else:
            return(self.getRandomAction())


    def getRandomAction(self):
        return(randint(0,self.agent.N_actions-1))




    def runEpisodes(self, N_eps=50):

        plt.close('all')
        fig, axes = plt.subplots(1,1,figsize=(8,6))
        axes.set_xlabel('episodes')
        axes.set_ylabel('steps per episode')

        steps_per_ep = []

        for i in range(N_eps):
            if i%int(N_eps/10) == 0:
                print('episode {}'.format(i))

            steps_per_ep.append(self.dynaQ(N_steps=10**4, show_plot=False, save_plot=False, N_plan_steps=0))


        axes.plot(steps_per_ep)
        plt.savefig(self.img_fname)




    def dynaQ(self, N_steps=100, show_plot=True, save_plot=True, N_plan_steps=5, quiet=False):


        if show_plot:
            self.showFig()

        self.initEpisode()
        s = self.getStateVec()

        for i in range(N_steps):

            if i%int(N_steps/10) == 0 and quiet==False:
                print('iteration {}, R_tot/i = {:.3f}'.format(i,self.R_tot_hist[-1]))

            self.updateEpsilon()

            a = self.epsGreedyAction(s)
            r, s_next = self.iterate(a)

            self.R_tot += r
            self.R_tot_hist.append(self.R_tot/(i+1))

            Q_cur = self.Q[s, a]
            Q_next = torch.max(self.Q[s_next]).detach()

            TD0_error = (r + self.params['gamma']*Q_next - Q_cur).pow(2).sum()

            self.optimizer.zero_grad()
            TD0_error.backward()
            self.optimizer.step()


            e = Experience(s, a, r, s_next)
            self.updateModel(e)


            if self.epFinished():
                N_steps_completed = i
                break

            if N_plan_steps>0:
                TD0_error = 0
                for j in range(N_plan_steps):

                    xp = self.experiences[randint(0,len(self.experiences)-1)]
                    r0, s_next0 = self.modelSimulatedExperience(xp.s, xp.a)
                    print('\n\nxp: {},\n r, s_next: {}'.format(xp,(r0,s_next0)))
                    Q_cur = self.Q[xp.s, xp.a]
                    Q_next = torch.max(self.Q[s_next0]).detach()
                    #TD0_error += (r0 + self.params['gamma']*Q_next - Q_cur)
                    TD0_error = (r0 + self.params['gamma']*Q_next - Q_cur).pow(2).sum()
                    self.optimizer.zero_grad()
                    TD0_error.backward()
                    self.optimizer.step()

                '''TD0_error = TD0_error.pow(2).sum()
                self.optimizer.zero_grad()
                TD0_error.backward()
                self.optimizer.step()'''



            s = s_next


            if show_plot:
                self.plotAll()
                self.fig.canvas.draw()

            N_steps_completed = i



        if save_plot:
            self.plotAll()
            plt.savefig(self.img_fname)


        return(N_steps_completed)






    def iterate(self, a):
        r, s_next = self.agent.iterate(a)
        return(r, s_next)


    def getStateVec(self):
        return(self.agent.getStateVec())


    def initEpisode(self):
        self.R_tot = 0
        self.R_tot_hist = [0]
        self.agent.initEpisode()
        self.params['epsilon'] = self.params['init_epsilon']


    def epFinished(self):
        return(self.agent.done())


    def createFigure(self):

        self.fig, self.axes = plt.subplots(3,3,figsize=(12,8))
        self.ax_state = self.axes[0,0]
        self.ax_state_params1 = self.axes[0,1]
        self.ax_state_params2 = self.axes[0,2]
        self.ax_state_params3 = self.axes[1,2]
        self.ax_state_params4 = self.axes[1,1]

        self.ax_wQ = self.axes[2,1]
        self.ax_wQ2 = self.axes[2,0]
        self.ax_theta_pi = self.axes[1,1]

        #self.ax_loc_vals = self.axes[2,0]
        self.ax_R_tot = self.axes[2,2]
        self.col_bar = None
        self.col_bar2 = None

        self.cm = LinearSegmentedColormap.from_list('my_cm', ['tomato','dodgerblue','seagreen','orange'], N=4)

        self.last_target_pos = None
        #self.plotAll()



    def plotAll(self):
        self.drawState()
        #self.plotStateParams()
        self.plotRtot()
        '''if self.params['features'] == 'DQN':
            self.plotWeights()'''
        #self.plotWeights()

    def plotRtot(self):
        self.ax_R_tot.clear()
        self.ax_R_tot.plot(self.R_tot_hist[:])


    def drawState(self):
        self.agent.drawState(self.ax_state)



    def showFig(self, block=False):
        plt.show(block=block)













































#
