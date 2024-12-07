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
    df = df.drop(df[df["Publication Year"] > 2024].index)
    df.reset_index(inplace=True)

    return df

def normalize_text(text):
    """Normalize text for comparison by removing punctuation, lowercasing, and trimming whitespace."""
    return re.sub(r'\W+', '', text.lower().strip())

def clean_dataframe(df):
    df["TitleNorm"] = df["Title"].map(normalize_text)
    df = df.drop_duplicates(subset=["TitleNorm"], keep="last")
    df.reset_index(inplace=True)
    return df

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
    # clean data
    df = clean_dataframe(df)

    models = [
        "facebook/bart-large-mnli", 
        "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    ]
    
    for model in models:
        modelstr = model.split("/")[1]
        # classify papers using NLP
        classifier = pipeline("zero-shot-classification", model=model, device=0, torch_dtype="auto")
        labels = [
            ["bedload transport"],
            ["mobile bed"],
            ["high performance computing enabled"],
            ["gpu accelerated"],
            ["distributed memory"],
            ["cpu multithreading"],
            ["two dimensional"],
            ["one dimensional"],
            ["deterministic"],
            ["stochastic"],
            ["finite volume method"],
            ["finite element method"],
            ["finite difference method"],
            ["numerical method development"],
            ["riverbed morphology"],
            ["hydrological modeling"], 
            ["ocean modeling"], 
            ["computational performance report"],
            ["validation of numerical method"],
            ["iber"],
            ["telemac"],
            ["hec-ras"]
        ]

        

        for label in labels:
            dataset = Dataset.from_pandas(df[['Abstract Note']])
            results = dataset.map(lambda x: process_batch(x, label, classifier), batched=True, batch_size=64)
            result_df = pd.DataFrame(results)
            # Flatten scores into the DataFrame and convert to float
            df[label[0]+"_"+modelstr] = result_df['scores'].map(lambda x: x if isinstance(x, float) else float(x[0]))

        df.to_csv("classified.csv")
        # Display the processed results