import altair as alt
from vega_datasets import data
import altair_viewer
import os
from preprocessing.presetting import local_temp_directory
alt.renderers.enable('altair_viewer')

source = data.cars()

chart = alt.Chart(source).mark_circle(size=60).encode(
    x='Horsepower',
    y='Miles_per_Gallon',
    color='Origin',
    tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
).interactive()

chart.save(os.path.join(local_temp_directory("wcph113"), "html_visual_altair.html"))

altair_viewer.display(chart)