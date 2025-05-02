import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class TrainingMonitor:
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.rewards = []
        self.epsilons = []
        self.frames = []
        
    def update(self, episode, reward, epsilon):
        self.rewards.append(reward)
        self.epsilons.append(epsilon)
        self.frames.append((self.rewards.copy(), self.epsilons.copy()))
        
    def animate(self, frame):
        rewards, epsilons = self.frames[frame]
        
        self.ax1.clear()
        self.ax1.plot(rewards, 'b-', linewidth=2)
        self.ax1.set_title("Episode Rewards")
        self.ax1.grid(True)
        
        self.ax2.clear()
        self.ax2.plot(epsilons, 'r-', linewidth=2)
        self.ax2.set_title("Exploration Rate (ε)")
        self.ax2.grid(True)
        
        plt.tight_layout()
        return self.ax1, self.ax2

    def save_animation(self, filename="outputs/training_progress.gif"):
        ani = FuncAnimation(
            self.fig, self.animate, frames=len(self.frames),
            interval=500, blit=False
        )
        ani.save(filename, writer=PillowWriter(fps=2))
        print(f"Saved training animation to {filename}")