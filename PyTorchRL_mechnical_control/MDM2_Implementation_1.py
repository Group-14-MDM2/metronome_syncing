
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class Pendulum_on_SlidingPlatform():
    def __init__(self,nPendulums,MassConfig,LConfig,DampingConfig ,g = 9.8):
        #Assume the pendulum have same masses/radius, Ill add more functions later
        self.nPendulums = nPendulums   #Int
        self.SliderM,self.M = MassConfig
        self.L = LConfig
        self.TotalM = self.SliderM + np.sum(self.M)
        self.g = g
        self.PlatformDampingCoef,self.PendulumDampingCoef = DampingConfig
        self.statedim = 2+2*self.nPendulums

        
    def equations(self, t, state,ControlForce = 0):
        #States contain Phase and location for each pendulum and slider
        # SliderX, SliderV, Phase0, PhaseVelocity0.... etc, solveivp only takes 1D array

        SliderX = state[0]
        SliderV = state[1]

        PhaseStates = []
        for i in range(self.nPendulums):
            idx = 2 + 2*i
            PhaseStates.append([state[idx], state[idx+1]])
    
        SliderA = 0
        # A = sum n / m - sum d 
        n = 0
        d = 0
        for i in range(self.nPendulums):
            Phase = PhaseStates[i][0]
            PhaseV = PhaseStates[i][1]
            m = self.M[i]
            L = self.L[i]
            n += m * ( self.g * np.sin(Phase) * np.cos(Phase) +  L * ( PhaseV**2 ) * np.sin(Phase) )
            d += m * (np.cos(Phase) ** 2 )

        n -= self.PlatformDampingCoef * SliderV - ControlForce

        SliderA = n / (np.sum(self.M) + self.SliderM - d )
           

        PhaseA = np.zeros(self.nPendulums)

        for i in  range(self.nPendulums):
            Phase = PhaseStates[i][0]
            PhaseV = PhaseStates[i][1]
            m = self.M[i]
            L = self.L[i]
            PhaseA[i] = -self.g/L *np.sin(Phase) - SliderA/L*np.cos(Phase) - self.PendulumDampingCoef /( m * L**2 ) * PhaseV



        dstate = np.zeros(self.statedim)
        dstate[0] = SliderV
        dstate[1] = SliderA

        for i in range(self.nPendulums):
            idx = 2 + 2*i
            dstate[idx] = PhaseStates[i][1]      
            dstate[idx+1] = PhaseA[i]            
    
        return dstate  

    def simulate(self, t_span, initial_conditions,controller = None):
        def ModelControlEquation(t, state):
            if controller != None:
                force = controller(state)
            else:
                force = 0

            return self.equations(t, state, ControlForce=force)

        sol = solve_ivp(ModelControlEquation, t_span, initial_conditions, 
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
    def get_order(self,state):
        angles = state[2::2] 
        order = np.sum(np.exp(1j * angles)) / self.nPendulums
        return order         
    def get_synchronize_time(self,sol,stable_therhold = 0.9):
        orders = np.array([self.get_order(sol.y[:, i], self.nPendulums) for i in range(sol.y.shape[1])])
        condition = orders >= stable_therhold
        if condition.any():
            return np.argmax(condition)
        else:
            return np.inf

# Run RL model with mechincal Pendulum_on_SlidingPlatform() as enviornment 

def run_episode(model,env:Pendulum_on_SlidingPlatform, t_max = 10,dt = 0.05):
    states, actions, rewards = [], [], []

    #Set random initial state 
    state = np.zeros(env.statedim)
    state[2::2] = np.random.uniform(-np.pi, np.pi, env.nPendulums)
    t = 0

    total_work = 0
    while t < t_max:
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            force_mean = model(state_tensor)

            force_dist = torch.distributions.Normal(force_mean, 0.1)
            action = force_dist.sample()
        
        f_val = action.item() * 1.0 #Multiply to get actual force
        sol = solve_ivp(
            fun=lambda _t, _y: env.equations(_t, _y, ControlForce=f_val),
            t_span=[t, t + dt],y0=state,method='RK45',rtol=1e-5,atol=1e-8  
        )# solveivp for a small dt, to get the next stage and reaction of the applied control force
        
        next_state = sol.y[:, -1]
        order = env.get_order(next_state)

        #Reward Equations: 
        alpha = 2.0 #Encourage Sychronize
        beta = 2.0 #Discourage energy spending, force**2 might be huge so we deccrease the factor
        theta = 5.0 #Time bonus, if the model sychronize realy they get a huge reward
        sychronize_therhold = 0.98

        r = alpha*np.abs(order) - beta * (f_val**2)

        states.append(state_tensor)
        actions.append(action)
        rewards.append(r)
        state = next_state
        t += dt
        total_work += np.abs(f_val * state[1] * dt)
        
        if np.abs(order) >= sychronize_therhold: # If sychronize give a huge bonus reward
            bonus = (t_max - t) * theta
            rewards[-1] += bonus
            break

    return states, actions, rewards

class ControlNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ControlNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, state):
        # state: [SliderX, SliderV, Phase1, PhaseV1, ...]
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        #Limit the force between [1,-1]
        force = torch.tanh(self.output_layer(x)) 
        return force

def train_model(env,model,optimizer,num_episodes = 200,savename = 'pendulum_ai_v1.pth'):
    best_sync_time = float('inf')
    for ep in range(num_episodes):
        torch.autograd.set_detect_anomaly(True)
        states, actions, rewards = run_episode(model, env)
        discounted_rewards = []
        cumulative_r = 0
        for r in reversed(rewards): # Discounted Returns,let the model consider future as an reward
            cumulative_r = r + 0.95 * cumulative_r
            discounted_rewards.insert(0, cumulative_r)
    
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        loss = 0
        for s, a, r in zip(states, actions, discounted_rewards):
            force_mean = model(s)
            dist = torch.distributions.Normal(force_mean, 0.1)
            log_prob = dist.log_prob(a)
            loss += -log_prob * r
    
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if ep % 10 == 0:
            print(f"Episode {ep}, Total Reward: {sum(rewards):.2f}, Final Order: {np.abs(env.get_order(states[-1].numpy())):.2f}")
        #print(f"Running {ep} episode")
        if ep % 100 == 0 and ep!= 0:
            torch.save(model.state_dict(), savename)
            print(f"Model saved to {savename}")

def test_model(state,env:Pendulum_on_SlidingPlatform ,model, t_max = 10,dt = 0.05):
    history = {'t': [], 'state': [], 'order': [], 'force': []}
    model.eval()
    #States were inputed externally, so differently trained model can experiment on same system, reduce performance differences due ot luck
    t = 0
    sychronize_therhold = 0.98

    while t < t_max:
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action = model(state_tensor)
        
        f_val = action.item() * 1.0
        sol = solve_ivp(
            fun=lambda _t, _y: env.equations(_t, _y, ControlForce=f_val),
            t_span=[t, t + dt],y0=state,method='RK45',rtol=1e-5,atol=1e-8  
        )# solveivp for a small dt, to get the next stage and reaction of the applied control force
        
        next_state = sol.y[:, -1]
        order = env.get_order(next_state)
        
        history['t'].append(t)
        history['state'].append(state.copy())
        history['order'].append(order)
        history['force'].append(f_val)
        state = next_state
        t += dt
    return history
    
def show_test_model(history,env:Pendulum_on_SlidingPlatform):
    t = np.array(history['t'])
    orders = np.array(history['order'])
    forces = np.array(history['force'])
    states = np.array(history['state'])

    dt = history['t'][1] - history['t'][0]
    velocities = np.array([s[1] for s in history['state']])
    total_work = np.sum(np.abs(forces * velocities)) * dt
    print(f"Total Work: {total_work:.2f} J")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)# plot order/sychronization progress
    axes[0].plot(t, orders, 'r-', linewidth=2)
    axes[0].axhline(y=0.9, color='k', linestyle='--', label='Threshold')
    axes[0].set_ylabel('Order Parameter |R|')
    axes[0].set_title('Synchronization Progress')
    axes[0].grid(True)
    for i in range(env.nPendulums): # plot phases of pendulums
        phases = states[:, 2 + 2*i]
        phases = (phases + np.pi) % (2 * np.pi) - np.pi
        axes[1].plot(t, phases, label=f'Pendulum {i+1}')
    axes[1].set_ylabel('Phase (rad)')
    axes[1].legend(loc='right')
    axes[1].grid(True)

    axes[2].plot(t, forces, 'g-')#Plot control force
    axes[2].set_ylabel('Control Force (N)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    





if __name__ == "__main__":
    np.random.seed(42)
    env = Pendulum_on_SlidingPlatform(nPendulums=5, MassConfig=[2.0, [0.5, 0.5, 0.5 ,0.5 ,0.5]],LConfig=[1,1,1,1,1],DampingConfig= [0.5, 0.5])
    model = ControlNetwork(input_dim=env.statedim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    state = np.zeros(env.statedim)
    state[2::2] = np.random.uniform(-np.pi, np.pi, env.nPendulums)
    History = test_model(state,env,model)
    show_test_model(History,env)
    train_model(env,model,optimizer)
    History = test_model(state,env,model)
    show_test_model(History,env)
    
    



    