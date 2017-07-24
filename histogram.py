import numpy as np
from numpy import histogram as nphistogram
from numpy import array, linspace, zeros, ones, ones_like
from numpy import sqrt, pi, exp
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, hist, plot, ion, axes, title
from matplotlib import animation as animation


class Kde:
    color = 'blue'
    lw = 1
    pause = False
    def __init__(self, dataset):
        self.fig = figure()
        self.data = dataset
        self.pause = False

    def KdeGaussin(self, showpts = False):
        data = self.data
        fig = self.fig


    def getKdeAni(self, totalframes=100, showpts= False, model = 2):
        data = self.data
        fig = self.fig
        l = len(data)
        if totalframes is None:
            totalframes = min(len(data)-1, 100)
        width = data.max() - data.min()
        left, right = data.min()-width,  data.max()+width
        ax = axes(xlim=(min(0, data.min()),1.1*data.max()),ylim=(-0.1,2))

        line, = ax.plot([], [], lw=2, color = 'blue')
        n, _bins, patches = hist(data, totalframes, normed=1, facecolor='green', alpha=0.0)
        if showpts:
            junk = plot(data,ones_like(data)*0.1,'go', color = 'black')
        if model == 0:
            hist_draw = True
            gauss_draw = False
        elif model == 1:
            hist_draw = False
            gauss_draw = True
        else:
            hist_draw = True
            gauss_draw = True

        def init():
            line.set_data([], [])
            return line,

        def gaussian(x,sigma,mu):
            # Why isn't this defined somewhere?! It must be!
            return (1/sqrt(2*pi*sigma**2)) *  exp(-((x-mu)**2)/(2*sigma**2))

        def onClick(event):
            global pause
            self.pause = not self.pause

        def animate(i):
            if (not self.pause):
                if hist_draw:
                    n, bins = nphistogram(data, i + 1, normed=False)
                    ax.set_ylim(0, 1.1 * n.max())
                    for j in range(len(n)):
                        rect, h = patches[j], n[j]
                        x2 = bins[j]
                        w = bins[j + 1] - x2
                        rect.set_height(h)
                        rect.set_x(x2)
                        rect.set_width(w)
                        rect.set_alpha(0.75)
                if gauss_draw:
                    numpts = len(data)
                    x1 = linspace(left, right, numpts)
                    dx = (right - left) / (numpts - 1)
                    y = zeros(l)
                    kernelwidth = .002 * width * (i + 1)
                    kernelpts = int(kernelwidth / dx)
                    kernel = gaussian(linspace(-3, 3, kernelpts), 1, 0)
                    for di in data:
                        center = di - left
                        centerpts = int(center / dx)
                        bottom = centerpts - int(kernelpts / 2)
                        top = centerpts + int(kernelpts / 2)
                        if top - bottom < kernelpts: top = top + 1
                        if top - bottom > kernelpts: top = top - 1
                        y[bottom:top] += kernel
                    ax.set_xlim( -1, 1.1 * data.max())
                    line.set_data(x1, y)
                    ax.set_ylim(min(0, y.min()), 1.1 * y.max())
                    title('ymin %s ymax %s' % (y.min(), y.max()))

            return line
        fig.canvas.mpl_connect('button_press_event', onClick)
        ani = animation.FuncAnimation(fig, animate, init_func=init, frames=100, repeat=False)
        plt.show()
        return ani