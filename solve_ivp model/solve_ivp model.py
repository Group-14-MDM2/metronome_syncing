
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Pendulum_on_SlidingPlatform():
    def __init__(self,nPendulums,g = 9.8):
        #Assume the pendulum have same masses/radius, Ill add more functions later
        self.nPendulums = nPendulums   #Int, Number of pendulums
        self.L = np.zeros(nPendulums) #Length of each pendulum, WIP
        self.M = np.zeros(nPendulums) #Mass of each pendulum, WIP
        self.SliderM = 100 #Platform mass
        self.TotalM = self.SliderM #Total Mass
        for i in range(self.nPendulums):
            self.L[i] = 10
            self.M[i] = 1
            self.TotalM += 1

        self.g = g

        self.statedim = 2+2*self.nPendulums
        
    def equations(self, t, state):
        #States contain Phase and location for each pendulum and slider
        # SliderX, SliderV, Phase0, PhaseVelocity0.... etc

        SliderX = state[0]
        SliderV = state[1]

        PhaseStates = []
        for i in range(self.nPendulums):
            idx = 2 + 2*i
            PhaseStates.append([state[idx], state[idx+1]])
    

        SliderA = 0
       
        #Formula for Platform acceleration
        DenoM = 0
        for i in range(self.nPendulums):
            Phase = PhaseStates[i][0]
            m = self.M[i]
            DenoM += m* np.square(np.sin(Phase))

        for i in range(self.nPendulums):
            Phase = PhaseStates[i][0]
            PhaseV = PhaseStates[i][1]
            m = self.M[i]
            L = self.L[i]
            SliderA += m * np.sin(Phase) * (self.g * np.cos(Phase) + L*PhaseV*PhaseV)/(self.SliderM + DenoM)

        PhaseA = np.zeros(self.nPendulums)

        #Equations for Phase acceleration of each pendulum
        for i in  range(self.nPendulums):
            Phase = PhaseStates[i][0]
            PhaseV = PhaseStates[i][1]
            m = self.M[i]
            L = self.L[i]
            PhaseA[i] = -self.g/L *np.sin(Phase) - SliderA/L*np.cos(Phase)

        #New state
        dstate = np.zeros(self.statedim)
        dstate[0] = SliderV
        dstate[1] = SliderA

        for i in range(self.nPendulums):
            idx = 2 + 2*i
            dstate[idx] = PhaseStates[i][1]      
            dstate[idx+1] = PhaseA[i]            
    
        return dstate  


    def simulate(self, t_span,Pendulum_Initial_position, Pendulum_Initial_velocity,Platform_initial_velocity):
        
        initial_conditions = []
        initial_conditions.append(0) #Platform initial position
        initial_conditions.append(Platform_initial_velocity)
        
        for i in range(len(Pendulum_Initial_position)):
            initial_conditions.append(Pendulum_Initial_position[i])
            initial_conditions.append(Pendulum_Initial_velocity[i])
        
        


        #Solveivp based on function Equation()
        sol = solve_ivp(self.equations, t_span, initial_conditions, 
                       method='RK45', dense_output=True)
        return sol


    def plot_trajectories(self, sol, figsize=(12, 8)):
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        t = sol.t
        slider_x = sol.y[0, :]
        slider_v = sol.y[1, :]
        

        axes[0].plot(t, slider_x, 'k-', label='Slider Position', linewidth=2)
        axes[0].set_ylabel('Slider Position')
        axes[0].legend()
        axes[0].grid(True)
        

        for i in range(self.nPendulums):
            idx = 2 + 2*i
            phase = sol.y[idx, :]
            phase_norm = (phase + np.pi) % (2 * np.pi) - np.pi
            axes[1].plot(t, phase_norm, label=f'Pendulum {i+1} Phase')
        
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Phase (rad)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        return fig, axes
    

    def plot_phase_space(self, sol, figsize=(12, 4)):
        if self.nPendulums == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            idx = 2
            phase = sol.y[idx, :]
            phase_v = sol.y[idx+1, :]
            ax.plot(phase, phase_v, 'b-', alpha=0.7)
            ax.set_xlabel('Phase')
            ax.set_ylabel('Angular Velocity')
            ax.set_title('Phase Space')
            ax.grid(True)
        else:
            fig, axes = plt.subplots(1, self.nPendulums, figsize=(4*self.nPendulums, 4))
            for i in range(self.nPendulums):
                idx = 2 + 2*i
                phase = sol.y[idx, :]
                phase_v = sol.y[idx+1, :]
                axes[i].plot(phase, phase_v, 'b-', alpha=0.7)
                axes[i].set_xlabel(f'Phase {i+1}')
                axes[i].set_ylabel(f'Angular Velocity {i+1}')
                axes[i].grid(True)
        
        plt.tight_layout()
        return fig


model = Pendulum_on_SlidingPlatform(2)
Pendulum_Initial_positon0 = [0.5,-0.5]
Pendulum_Initial_velocity0 = [0,0]
Platform_initial_velocity0 = 0


solution = model.simulate((0,500), Pendulum_Initial_positon0, Pendulum_Initial_velocity0,Platform_initial_velocity0 )
model.plot_trajectories(solution)