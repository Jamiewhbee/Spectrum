# here will go function from the inpiration.ipynb
import matplotlib as mpl
import math
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy

from .dataAnalysis import *
#to do:
# change architecture
#1 class called plot
#different functions


class plot:
    def __init__(self,df):
        self.df = df
        #removing duplicate entries
        self.df = self.df.groupby(by = self.df.index.name).mean()    
    def add_caption(self, ax, caption, x_frac, y_frac, color="k", fontsize=10.0):
        """add caption function: adds a caption to the set of axes specified

        Args:
            ax (object): set of axes on which the caption is to be added
            caption (str): text to be added as caption
            x_frac (float): fractional x position of caption placement
            y_frac (float): fractional y position of caption placement
            color (str, optional): color of text in caption. Defaults to "k".
            fontsize (int, optional): size of font in caption text. Defaults to 10.
        """        

        #generating positions of figure
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_pos = x_min + x_frac * (x_max - x_min)
        y_pos = y_min + y_frac * (y_max - y_min)
        #adding caption
        ax.text(x_pos, y_pos, caption, fontsize=fontsize, color=color, fontweight="bold", fontfamily = 'serif')
        
  
    def false_colour(self, fig, ax, I_min = 0.9, I_max = 1.1, e_min = 50, e_max = 250, CB = True, label = True, units = "meV",cmap ="Blues_r"):
        """Plots false colour plot onto specified fig,ax.

        Args:
            fig (object): figure onto which the data is to be plotted
            ax (object): set of axes onto which the data is to be plotted
            I_min (float, optional): minimum value of intensity to be plotted. Defaults to 0.9.
            I_max (float, optional): maximum value of intensity to be plotted. Defaults to 1.1.
            e_min (float, optional): lower bound of x-axis. Defaults to 50.
            e_max (float, optional): upper bound of y axis. Default to 250.
            CB (bool, optional): Specifies the presence of a colour bar. Defaults to True.
            label (bool, optional): specifies the presence of a y-axis label. Defaults to True.
            units (str, optional): specifies the units of the y-axis - meV/cm-1/THz. Defaults to "meV".
            cmap (str, optional): colour map used in false colour plot. Defaults to "Blues_r"
        """        
        def ColorBarFormat(image, nBin = 4, size = "2%", pad = "2%", formatter = "%.2f"):
            """formats color bar

            Args:
                image (_type_): plot onto which the colour bar is to be added
                nBin (int, optional): number of ticks included on colour bar. Defaults to 4.
                size (str, optional): relative size of colour bar. Defaults to "2%".
                pad (str, optional): relative distance from colour bar to graph. Defaults to "2%".
                formatter (str, optional): ?. Defaults to "%.2f".
            """

            ax_divider = make_axes_locatable(ax)
            #needs to delete axes created by previous executions of ax_divider
            cax0 = ax_divider.append_axes("top", size = size, pad = pad)
            cb0 = fig.colorbar(image, cax = cax0, orientation = "horizontal")
            cb0.locator = ticker.MaxNLocator(nbins = nBin)
            cb0.update_ticks()
            cb0.ax.tick_params(direction="in")
            cax0.xaxis.set_ticks_position("top")
            cax0.xaxis.set_major_formatter(FormatStrFormatter(formatter))
        
            """Inputing data from pandas dataframe into numpy arrays.
            """
        #Changing units of index column
        self.df = Treatment.Change_units(self.df,units)

        #Cropping the dataframe according to the specified x range: [e_min, e_max].
        if self.df.index.max() < e_max:
            e_max = self.df.index.max()
        if self.df.index.min() > e_min:
            e_min = self.df.index.min()

        
        cropped_df = self.df.loc[e_min:e_max]

        #Creating numpy arrays for axes of plot
        x_data = cropped_df.columns.values.astype(float)
        y_data = cropped_df.index.values
        z_data = cropped_df.values
        
        #Setting up ticks on graph
        ax.tick_params(axis = "both", which = "major", direction="in")
        ax.tick_params(axis="both", which="minor", direction="in")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_ticks_position("both")

        

        #Plotting data from numpy arrays onto pcolormesh        
        CM1 = ax.pcolormesh(x_data, y_data, z_data, cmap=cmap, vmin=I_min, vmax=I_max)
        ax.set(xlabel="Magnetic Field (T)")

        #Adding label
        if label == True:
            ax.set(ylabel=f"Energy ({units})")
        else:
            ax.set(ylabel="")
            ax.set_yticklabels([])

        #Adding color bar
        if CB:
            ColorBarFormat(CM1)
        else:
            return

    def stacked_plot(self, fig, ax, e_min = 70.0, e_max = 250.0, B_maj_int = 2.0, B_min_int = 0.25, e_maj_int = 25, e_min_int = 5,units = 'meV', C_HL = '#011f4b', C_Base = '#005b96',alpha = 0.8, label = True, step = 0.01, grad_colours = False, cmap = "magma", I_min = 0.0, I_max = 1.0, zeroT_baseline = 1.0):
        """Plots stacked plot onto specified fig,ax

        Args:
            fig (object): figure on which the data is to be plotted
            ax (object): set of axes on which the data is to be
            e_min (float, optional): Lower bound of x axis on graph. Defaults to 70.0.
            e_max (float, optional): Upper bound of x axis on graph. Defaults to 250.0.
            B_maj_int (float, optional): Major interval in magnetic field of stacked plot. Defaults to 2.0.
            B_min_int (float, optional): Minor interval in magnetic field of stacked plot. Defaults to 0.25.
            e_maj_int (int, optional): Major interval of x ticks. Defaults to 25.
            e_min_int (int, optional): Minor interval of x ticks. Defaults to 5.
            units (str, optional): Units of quantity on x axis. Defaults to 'meV'.
            C_HL (str, optional): Colour of major interval in magnetic field plots. Defaults to '#011f4b'.
            C_Base (str, optional): Colour of minor interval in magnetic field plots. Defaults to '#005b96'.
            alpha (float, optional): Opacity of plots. Defaults to 0.8.
            label (bool, optional): Specifies the presence of stacked plot labels. Defaults to True.
            step (float, optional): Specifies the separation on the plot of successive lines.
            grad_colours (Boolean, optional): Specifies whether for stacked plots to follow colour map. Defaults to False.
            cmap (str, optional): Colour map to be used if grad_colours == True. Defaults to 'magma'.
            I_min (float, optional): Lower bound for colour map range. Defaults to 0.0. Required to be [0,1].
            I_max (float, optional): Upper bound for colour map range. Defaults to 1.0. Required to be [0,1].
            zeroT_baseline (float, optional): Specifies the y axis value for the 0T baseline value.
            
        """        
        #Changing units of index column
        self.df = Treatment.Change_units(self.df,units)

        #Generating offset vector
        offset=  np.arange(step,step*np.size(self.df,axis= 1)+step, step)

        #Applying offset vector to dataframe
        data_off = self.df + offset

        #Inserting one for nice straight line at the beginning
        data_off.insert(0, 0.0, zeroT_baseline)

        #Ensuring bounds specified do not lie outside the range of the dataframe.
        #If they are this can interfere with the interpolation

        if self.df.index.max() < e_max:
            e_max = self.df.index.max()
        if self.df.index.min() > e_min:
            e_min = self.df.index.min()

        
        B_max = self.df.columns.max()
        
        #Major intervals
        B_HL = np.round_(np.arange(0, B_max + B_maj_int, B_maj_int),6)

        #Minor intervals
        B = np.round_(np.arange(0, B_max + B_min_int, B_min_int),6)

        #axis and tick formatting
        ax.set(ylabel='$T_B/T_0$', xlabel=units, xlim=(e_min,e_max))
        ax.tick_params(axis='both',which='major',direction='in')
        ax.tick_params(axis='both',which='minor',direction='in')
        ax.xaxis.set_major_locator(MultipleLocator(e_maj_int))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(e_min_int))
        ax.xaxis.set_ticks_position('both')

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_ticks_position('both')

        #plotting major intervals in magnetic field

        #standard stacked plot binary colour scheme
        if grad_colours != True:

            #plotting major intervals in magnetic field
            for i in B_HL:
                ax.plot(data_off.loc[e_min:e_max,i],color=C_HL,alpha=alpha,zorder=1)
            
                #Adding label to each major plot
                if label:
                    ax.text(e_max + 5,data_off.loc[e_min:e_max,i].iloc[-1],f'{i:.1f}T',color=C_HL,size=12)
        
            #plotting minor intervals in magnetic field
            for i in B:
                ax.plot(data_off.loc[e_min:e_max,i],color=C_Base,alpha=alpha,zorder=-1)

        #colour gradient for successive stacked plots
        if grad_colours:
            #generating colour map for lines
            colors = np.linspace(I_min,I_max,len(B))
            color_map = mpl.cm.get_cmap(cmap)

            #plotting stacked plot lines
            for j,i in enumerate(B):
                ax.plot(data_off.loc[e_min:e_max,i],color = color_map(colors[j]),alpha=alpha,zorder=-1)

            #generating colour map for label text
            txt_colors = np.linspace(I_min,I_max,len(B_HL))
            txt_color_map = mpl.cm.get_cmap(cmap)

            #adding labels to major intervals
            for j,i in enumerate(B_HL):
                ax.text(e_max + 5,data_off.loc[e_min:e_max,i].iloc[-1],f'{i:.1f}T',color=txt_color_map(txt_colors[j]),size=12)
            

        
    

    def simple_plot(self,fig,ax,x_min = 0.0, x_max = 1000.0, y_min = -2.0, y_max = 4.0, y_label = 'Intensity', cmap = 'magma', I_min = 0.0, I_max = 1.0):
        """Plots 1D plot, suitable for optical data with discrete intervals ie separate temperatures

        Args:
            fig (object): fig on which data is to be plotted
            ax (object): set of axes on which data is to be plotted.
            x_min (float, optional): Lower bound of x axis. Defaults to 0.0.
            x_max (float, optional): Upper bound of x axis. Defaults to 1000.0.
            y_min (float, optional): Lower bound of y axis. Defaults to -2.0.
            y_max (float, optional): Upper bound of y axis. Defaults to 4.0.
            y_label (str, optional): Label to be displayed on y axis. Defaults to 'Intensity'.
            cmap (str, optional): Colour map to be used for different plots. Defaults to 'magma'.
            I_min (float, optional): Lower bound to be used in colour map. Defaults to 0.0. Required to be [0,1].
            I_max (float, optional): Upper bound to be used in colour map. Defaults to 1.0. Required to be [0,1].

        Returns:
            _type_: _description_
        """
        #units conversion for secondary axis
        def w_to_meV(x):
            return (10**2 * x * 2 * math.pi * scipy.constants.hbar * scipy.constants.c) / (scipy.constants.e * 10**-3)
        
        def meV_to_w(x):
            return (scipy.constants.e * 10**-3* x) / (10**2 * 2 * math.pi * scipy.constants.hbar*scipy.constants.c)
        
        #prepping data into cm-1 for consistent processing
        self.df = Treatment.Change_units(self.df,'cm-1')

        #axis formatting
        ax.set(ylabel=y_label, xlabel=r'Wavenumber (cm$^{-1}$)', xlim=(x_min, x_max), ylim =(y_min,y_max))
        ax.tick_params(axis='both',which='major',direction='in')
        ax.tick_params(axis='both',which='minor',direction='in')
        
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.xaxis.set_ticks_position('both')

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.yaxis.set_ticks_position('both')

        #secondary axis formatting
        secax = ax.secondary_xaxis('top', functions=(w_to_meV, meV_to_w))
        secax.set_xlabel('Photon energy (mev)')

        #plotting data

        #generating colour map
        colors = np.linspace(I_min,I_max,len(self.df.columns))
        color_map = mpl.cm.get_cmap(cmap)

        #plotting data
        for i,temperature in enumerate(self.df.columns):
            ax.plot(self.df.index, self.df[temperature], label = (str(temperature) + 'K'), color = color_map(colors[i]))
        fig.legend()













   




    
    
    


