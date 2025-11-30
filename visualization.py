# visualization.py
import matplotlib.pyplot as plt
import numpy as np
import imageio

def plot_line(x, data, title="", xlabel="Distance (m)", ylabel="Concentration"):
    """Plot a concentration profile along the channel at one time."""
    plt.figure()
    plt.plot(x, data, '-o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    st.pyplot(plt)  # If using inside Streamlit
    plt.clf()

def animate_profile(x, profiles, fps=5):
    """
    Create an animated GIF of property profiles over time.
    `profiles` is a list of concentration arrays (time snapshots).
    """
    images = []
    for i, C in enumerate(profiles):
        plt.figure()
        plt.plot(x, C, 'b-')
        plt.ylim(min(profiles[0]), max(profiles[0])*1.1)
        plt.title(f"Time step {i}")
        plt.xlabel("Distance (m)")
        plt.ylabel("Concentration")
        plt.grid(True)
        # Save to temporary buffer
        fname = f"_temp_frame_{i}.png"
        plt.savefig(fname)
        plt.close()
        images.append(imageio.imread(fname))
    imageio.mimsave('profile_animation.gif', images, fps=fps)
    return 'profile_animation.gif'
