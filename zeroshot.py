import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True


df = pd.read_csv("classified.csv")

df["BLT"] = df["bed load transport"].map(lambda x: x.replace("[", "").replace("]", "")).astype(float) > 0.85
df["HPC"] = df["high performance computing enabled"].map(lambda x: x.replace("[", "").replace("]", "")).astype(float) > 0.85
df["2D"] = df["two dimensional"].map(lambda x: x.replace("[", "").replace("]", "")).astype(float) > 0.85
df["FVM"] = df["finite Volume method"].map(lambda x: x.replace("[", "").replace("]", "")).astype(float) > 0.85
df["FEM"] = df["finite element method"].map(lambda x: x.replace("[", "").replace("]", "")).astype(float) > 0.85
df["FDM"] = df["finite difference method"].map(lambda x: x.replace("[", "").replace("]", "")).astype(float) > 0.85
df["NM"] = df["numerical method development"].map(lambda x: x.replace("[", "").replace("]", "")).astype(float) > 0.85

labels = ["BLT","HPC", "2D", "FVM", "FEM", "FDM", "NM"]
custom_colors = ['#ff9999', '#66b3ff', '#99ff99']

fig, ax = plt.subplots()

trues = []
falses = []
values = None
for label in labels:
    classes, values = np.unique(df[label], return_counts=True)
    trues.append(100*values[1]/np.sum(values))
    falses.append(100*values[0]/np.sum(values))

z = [x for _, x in sorted(zip(trues, labels), reverse=True)]

#plt.bar(labels, falses, color=custom_colors[0])
plt.bar(z, sorted(trues, reverse=True), color=custom_colors[1])
plt.grid()
plt.xlabel("Topic Abbreviation")
plt.ylabel("Percentage of papers")
plt.text(5.5, 60, "Total papers: "+str(np.sum(values)), size=10,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(0., 0.5, 0.5),
                   fc=(1., 1, 1),
                   )
         )
plt.savefig("overall.png")
plt.close()

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=trues, 
    theta=labels,
    name="Total papers: 92"
))
fig.update_layout(
            polar=dict(
                radialaxis=dict(
                visible=True
                ),
            ),
            font=dict(
                size=18,  # Set the font size here
            ),
            legend={
                "x": 0.3,
                "xref": "container",
                "yref": "container",
                "orientation": "h"
            },
            showlegend=True
        )
fig.write_image("radar_performance.png",width=800, height=600, scale=1.5)
