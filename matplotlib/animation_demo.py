

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def line_plot():
    fig, ax = plt.subplots()

    x = np.arange(0, 2*np.pi, 0.01)
    line, = ax.plot(x, np.sin(x))


    def animate(i):
        line.set_ydata(np.sin(x + i / 50))  # update the data.
        return line,


    ani = animation.FuncAnimation(
        fig, animate, interval=20, blit=True, save_count=50)

    # To save the animation, use e.g.
    #
    # ani.save("movie.mp4")
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)

    plt.show()

def tmp():
    vector = [1,2]

    fig, ax = plt.subplots()
    arr = np.array([[1,2], [2,4]])

    new_matrix, = ax.plot(arr, )
    # x = np.arange(0, 2*np.pi, 0.01)
    # line, = ax.plot(x, np.sin(x))

    def animate(i):


        line.set_ydata(np.sin(x + i / 50))  # update the data.
        return line,

    ani = animation.FuncAnimation(
        fig,
    )
