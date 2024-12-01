import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import re
plt.rcParams['axes.axisbelow'] = True

from datasets import Dataset
from transformers import pipeline, BatchEncoding

def unify_databases(sources):
    source_dfs = []

    for source in sources:
        tmpdf = pd.read_csv("Monografia-"+source+".csv")
        tmpdf["Source"] = source
        source_dfs.append(tmpdf)
    
    df = pd.concat(source_dfs, axis=0)
    df = df.drop(df[df["Publication Year"] < 2000].index)
    df.reset_index(inplace=True)

    return df

def normalize_text(text):
    """Normalize text for comparison by removing punctuation, lowercasing, and trimming whitespace."""
    return re.sub(r'\W+', '', text.strip())

def clean_dataframe(df):
    df["TitleNorm"] = df["Title"].map(normalize_text)
    df = df.drop_duplicates(subset=["TitleNorm"], keep="last")
    df.reset_index(inplace=True)
    return df

def gensim_tokenize_data(df):
    df["Abstract Note"] = df["Abstract Note"].map(str)
    df["gensim_tokenized_titles"] = df["Title"].map(lambda x: simple_preprocess(x.lower(), deacc=True))
    df["gensim_tokenized_abstracts"] = df["Abstract Note"].map(lambda x: simple_preprocess(x.lower(), deacc=True))
    return df


def graph_distribution(df, name=None):
    plt.hist(df["Publication Year"],bins=9)
    plt.grid(alpha=0.5)
    plt.xlabel("Year")
    plt.ylabel("Paper count")
    plt.xticks(np.arange(2007,2025,2))
    plt.savefig("yeardistribution"+name+".png", dpi=130)
    plt.close()

def graph_sources(df, name=None):
    sources_df = df.groupby("Source").count()["Title"]
    plt.figure(figsize=(11,5))
    ax = sources_df.plot(kind="barh")
    ax.set_xlabel("Paper counts")
    plt.grid(alpha=0.5)
    plt.savefig("barplot"+name+".png")
    plt.close()

    # Function to process abstracts in batches
def process_batch(batch, labels, classifier):
    # Apply the classifier to a batch of abstracts
    abstracts = batch['Abstract Note']
    results = classifier(abstracts, candidate_labels=labels)
    return {
        'scores': [result['scores'][0] for result in results],  # Extract first score
        'labels': [result['labels'] for result in results]
    }

if __name__ == "__main__":
    sources = ["Google-Scholar", "Web-of-Science", "IEEE", "Scopus"]
    df = unify_databases(sources)
    #graph_sources(df, name="_raw")
    #graph_distribution(df, name="_raw")
    # clean data
    df = clean_dataframe(df)
    #graph_distribution(df, name="_processed")
    #graph_sources(df, name="_processed")

    # classify papers using NLP
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", device=0, torch_dtype="auto")
    labels = [
        ["bed load transport"],
        ["high performance computing enabled"],
        ["two dimensional"],
        ["finite Volume method"],
        ["finite element method"],
        ["finite difference method"],
        ["numerical method development"],
        ["riverbed morphology"],
        ["general hydrological modeling"], 
        ["ocean modeling"], 
        ["presents computational performance reports"],
        ["performs validation of numerical solution"]
    ]

    for label in labels:
        dataset = Dataset.from_pandas(df[['Abstract Note']])

        # Apply batch processing
        # Results are better when it's binary processing instead of Multi-Label classification
        batch_size = 64  # You can adjust the batch size based on available GPU memory
        results = dataset.map(lambda x: process_batch(x, label, classifier), batched=True, batch_size=batch_size)
        # Convert the results back to a pandas DataFrame for easier analysis
        result_df = pd.DataFrame(results)        
        df[label[0]] = result_df["scores"]

    df.to_csv("classified.csv")
    # Display the processed results

    df["BLT"] = df["bed load transport"].map(lambda x: x.replace("[", "").replace("]", "")).astype(float) > 0.85
    df["HPC"] = df["high performance computing enabled"].map(lambda x: x.replace("[", "").replace("]", "")).astype(float) > 0.85
    df["2D"] = df["two dimensional"].map(lambda x: x.replace("[", "").replace("]", "")).astype(float) > 0.85
    df["FVM"] = df["finite Volume method"].map(lambda x: x.replace("[", "").replace("]", "")).astype(float) > 0.85
    df["FEM"] = df["finite element method"].map(lambda x: x.replace("[", "").replace("]", "")).astype(float) > 0.85
    df["FDM"] = df["finite difference method"].map(lambda x: x.replace("[", "").replace("]", "")).astype(float) > 0.85
    df["NMD"] = df["numerical method development"].map(lambda x: x.replace("[", "").replace("]", "")).astype(float) > 0.85
    df["CPR"] = df["presents computational performance reports"].map(lambda x: x.replace("[", "").replace("]", "")).astype(float) > 0.85
    df["VNS"] = df["performs validation of numerical solution"].map(lambda x: x.replace("[", "").replace("]", "")).astype(float) > 0.85

    labels = ["BLT", "HPC", "2D", "FVM", "FEM", "FDM", "NMD", "CPR", "VNS"]
    custom_colors = ['#ff9999', '#66b3ff', '#99ff99']

    fig, ax = plt.subplots()

    trues = []
    falses = []
    values = None
    for label in labels:
        classes, values = np.unique(df[label], return_counts=True)
        trues.append(100*values[1]/np.sum(values))

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

    """    fig = go.Figure()
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
    """