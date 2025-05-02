import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter

class NetworkVisualizer:
    def __init__(self, topology):
        self.topology = topology
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.frames = []
        
    def add_frame(self, path=None):
        self.frames.append(path)
    
    def animate(self, frame):
        self.ax.clear()
        pos = nx.spring_layout(self.topology)
        
        # Draw all elements
        nx.draw_networkx_nodes(self.topology, pos, ax=self.ax, node_size=500)
        nx.draw_networkx_edges(self.topology, pos, ax=self.ax, width=2)
        nx.draw_networkx_labels(self.topology, pos, ax=self.ax)
        
        # Highlight path
        if self.frames[frame]:
            path_edges = list(zip(self.frames[frame][:-1], self.frames[frame][1:]))
            nx.draw_networkx_edges(
                self.topology, pos, edgelist=path_edges,
                edge_color='r', width=4, ax=self.ax
            )
        
        self.ax.set_title(f"Frame {frame}: {'→'.join(self.frames[frame]) if self.frames[frame] else 'Exploring'}")
        return self.ax

    def save_animation(self, filename="outputs/sdwan_routing.gif"):
        ani = FuncAnimation(
            self.fig, self.animate, frames=len(self.frames),
            interval=500, blit=False
        )
        ani.save(filename, writer=PillowWriter(fps=2))
        print(f"Saved animation to {filename}")