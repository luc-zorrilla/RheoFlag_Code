# Plotting
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import copy
import re
# pio.templates.default = "custom_template"

dark_purple_scale = ["#110022", "#2E0141", "#56065D", "#770D6A", "#8D1866", "#9A1D8A", "#9B22A6", "#8928B1", "#7533BB"]
dark_blue_scale = ["#031D23", "#082C44", "#103062", "#1B2D7D", "#2A2895", "#2C42A1", "#3162AC", "#3784B6", "#3DA8C0"]

# Colorblind palette:
cb_orange = '#EB6123' # works with cb_purple, white, black
cb_dark_orange = '#c64a12'
cb_purple = '#512888' # works with cb_orange, white
cb_dark_purple = '#41206d'
cb_light_cassis = '#9384b1' # works with white, black
purple = "#b1529e"
cb_blue = "#0b7fa5" # Works with white, black
cb_dark_blue = "#096684"
cb_pink = "#DB4C77"
cb_dark_pink = "#c42857"
cb_red = "#e5213a" # works with white, black
cb_dark_red = "#B31529" # works with white but not so well with black (3.06 contrast)

# Calculations
import numpy as np
from datetime import datetime

def invert_hex_color(hex_color):
    # Remove the '#' if present
    hex_color = hex_color.lstrip('#')

    # Convert the hex color to RGB components
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Invert the RGB values
    inverted_r = 255 - r
    inverted_g = 255 - g
    inverted_b = 255 - b

    # Convert back to hex and return as a string, ensuring 2 digits for each
    return '#{:02X}{:02X}{:02X}'.format(inverted_r, inverted_g, inverted_b)

def invert_rgb_string(rgb_string):
    # Use a regular expression to extract the r, g, b values from the input string
    match = re.match(r'rgb\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)', rgb_string)
    
    if not match:
        raise ValueError("Input string must be in the format 'rgb(r,g,b)'")

    # Extract the r, g, b values as integers
    r, g, b = map(int, match.groups())

    # Ensure the values are within the valid range [0, 255]
    if not all(0 <= value <= 255 for value in (r, g, b)):
        raise ValueError("RGB values must be between 0 and 255")

    # Invert each RGB component
    inverted_r = 255 - r
    inverted_g = 255 - g
    inverted_b = 255 - b

    # Return the inverted color in the 'rgb(r,g,b)' format
    return f'rgb({inverted_r},{inverted_g},{inverted_b})'

reverted_Tealrose = [invert_rgb_string(px.colors.diverging.Tealrose[k]) for k in range(len(px.colors.diverging.Tealrose))]

def switch_log(self):
    """ Adds a button to switch between linear and log scale for a given figure.
    Returns the figure with the added button. """

    updatemenus = list([
        dict(active=1,
            buttons=list([
                dict(label='Log Scale',
                    method='update',
                    args=[{'visible': [True]},
                        {'title': 'Log scale',
                            'yaxis': {'type': 'log'}}]),
                    
                dict(label='Linear Scale',
                    method='update',
                    args=[{'visible': [True]},
                        {'title': 'Lin scale',
                            'yaxis': {'type': 'linear'}}])]     
                            ))])

    layout = dict(updatemenus=updatemenus, title='Linear scale')
    self.update_layout(layout)
go.Figure.switch_log = switch_log

# Number formatting

# Define function for string formatting of scientific notation
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with:
    - specified number of significant decimal digits. If <0, do not return the prefactor. 
    - precision (number of decimal digits to show). If None, take decimal_digits.
    - the exponent to be used can also be specified explicitly.
    """

    if num == 0:
        return r"0"
    else:
        if exponent is None:
            exponent = int(np.floor(np.log10(abs(num))))
        coeff = np.round(num / float(10**exponent), decimal_digits)
        if precision is None:
            precision = decimal_digits

        if (decimal_digits >=0) & (coeff != 1):
            if exponent == 0:
                return r"{0:.{1}f}".format(coeff, precision)
            elif exponent == 1:
                return r"{0:.{1}f}\cdot10".format(coeff, precision)
            else:
                return r"{0:.{2}f}\cdot10^{{\displaystyle {1:d}}}".format(coeff, exponent, precision)
        # Show powers of 10 without prefactor ('coeff') 
        else:
            if exponent == 0:
                return r"1"
            elif exponent == 1:
                return r"10"
            else:    
                return r"10^{{\displaystyle {0:d}}}".format(exponent)
                

##################
# Plotly templates
##################

# Basic template used for visualization in vs code
custom_template = copy.deepcopy(pio.templates["plotly_dark"])
custom_layout = dict(
    plot_bgcolor = 'rgba(0, 0, 0, 0)',
    paper_bgcolor = 'rgba(0, 0, 0, 0)',
    font = dict(size = 10, color = 'black'),
    autosize = False,
    width = 1000,
    height = 500,
    xaxis = dict(showgrid = False, zeroline = False, showline = False),
    yaxis = dict(showgrid = False, zeroline = False, showline = False),
    showlegend = False,
    )
custom_template['layout'] = custom_layout
pio.templates["custom_template"] = custom_template

# Figure template
figure_template = copy.deepcopy(custom_template)
figure_layout = dict(
    # Background and dimensions
    plot_bgcolor = 'rgba(0, 0, 0, 0)',
    paper_bgcolor = 'rgba(0, 0, 0, 0)',
    autosize = False,


    # margin = dict(l = 400, r = 100, t = 100, b = 400),
    # width = 1000, # GRAPH size
    # height = 1000, # GRAPH size
    
    # Font
    font = dict(size = 40, color = 'black', family = 'Roman'),
    title_font = dict(size = 40, color = 'black', family = 'Roman'),
    legend_title_font = dict(size = 40, color = 'black', family = 'Roman'),
    # Axes
    xaxis = dict(
        showgrid = False, 
        zeroline = False,
        # Axis line
        showline = True,
        linewidth = 6, 
        linecolor = 'rgb(0,0,0,1)',
        # Ticks
        ticks="outside", 
        tickcolor = 'rgba(0,0,0,1)', 
        tickwidth = 6, 
        ticklen = 12,
        tickfont = dict(size = 40, color = 'black', family = 'Roman'),
        title_font = dict(size = 40, color = 'black', family = 'Roman'),     
        ),
    yaxis = dict(
        showgrid = False, 
        zeroline = False,
        # Axis line
        showline = True,
        linewidth = 6, 
        linecolor = 'rgb(0,0,0,1)',
        # Ticks
        ticks="outside", 
        tickcolor = 'rgba(0,0,0,1)', 
        tickwidth = 6, 
        ticklen = 12,
        tickfont = dict(size = 40, color = 'black', family = 'Roman'),
        title_font = dict(size = 40, color = 'black', family = 'Roman'),
        ),
    # Legend
    showlegend = True,
    legend = dict(
        bgcolor = 'rgba(255,255,255,0)',
        itemwidth = 50,
        ),
    )
figure_template['layout'] = figure_layout
pio.templates["figure_template"] = figure_template


if __name__ == '__main__':

    # Set default template to custom plotly template
    pio.templates.default = "figure_template"

    ## Figure without axis and labels: plot has size (width, height) in that case
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        y=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    ))

    fig.update_xaxes(showline = False, ticks = "", showticklabels = False)
    fig.update_yaxes(showline = False, ticks = "", showticklabels = False)

    fig.update_layout(
        autosize=False,
        width=500,
        height=500,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0,
        ),
        paper_bgcolor="LightSteelBlue",
        showlegend = False,
    )
    # fig.vs_show()
    fig.write_image(temp_dir + 'figure_noaxis_nolabel_nomargin.svg')

    ## Figure with margin: plot has size (width, height) - (margin_x, margin_y)
    ## With margin_x = margin_l + margin_r, margin_y = margin_t + margin_b
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        y=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    ))

    fig.update_xaxes(showline = False, ticks = "", showticklabels = False)
    fig.update_yaxes(showline = False, ticks = "", showticklabels = False)

    fig.update_layout(
        autosize=False,
        margin=dict(
            l=10,
            r=10,
            b=20,
            t=20,
            pad=0,
        ),        
        width=500 + 10 + 10,
        height=500 + 20 + 20,

        paper_bgcolor="LightSteelBlue",
        showlegend = False,
    )
    # fig.vs_show()
    fig.write_image(temp_dir + 'figure_noaxis_nolabel_margin.svg')

    ## Figure with axis and labels: plot has size (width, height) - (margin_x, margin_y)
    ## With margin_x = margin_l + margin_r, margin_y = margin_t + margin_b
    ## IF margins are large enough: this is essential, otherwise the plot will be compressed.
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        y=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    ))

    fig.update_xaxes(showline = True, ticks = "outside", showticklabels = True)
    fig.update_yaxes(showline = True, ticks = "outside", showticklabels = True)

    fig.update_layout(
        autosize=False,
        margin=dict(
            l=200,
            r=200,
            b=200,
            t=200,
            pad=0,
        ),        
        width=500 + 200 + 200,
        height=500 + 200 + 200,

        paper_bgcolor="LightSteelBlue",
        showlegend = False,
    )
    # fig.vs_show()
    fig.write_image(temp_dir + 'figure_axis_label_margin.svg')