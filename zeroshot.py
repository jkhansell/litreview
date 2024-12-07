import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

plt.rcParams['axes.axisbelow'] = True
custom_colors = ['#ff9999', '#66b3ff', '#99ff99']

def graph_distribution(df):
    plt.figure(figsize=(6,4))
    plt.hist(df["Publication Year"],bins=8)
    plt.grid(alpha=0.5)
    plt.xlabel("Year")
    plt.ylabel("Paper count")
    plt.savefig("yeardistribution_processed.png", dpi=130)
    plt.close()

def graph_sources(df):
    sources_df = df.groupby("Source").count()["Title"]
    plt.figure(figsize=(12,4))
    ax = sources_df.plot(kind="barh")
    ax.set_xlabel("Paper counts")
    plt.grid(alpha=0.5)
    plt.savefig("barplot_processed.png")
    plt.close()

def plot_class_distribution(df, models, labels):
    for model in models:

        df["BLT"+"_"+model] = df["bedload transport"+"_"+model].astype(float) > 0.875
        df["MB"+"_"+model] = df["mobile bed"+"_"+model].astype(float) > 0.875
        df["HPC"+"_"+model] = df["high performance computing enabled"+"_"+model].astype(float) > 0.875
        df["GPU"+"_"+model] = df["gpu accelerated"+"_"+model].astype(float) > 0.875
        df["CPU"+"_"+model] = df["cpu multithreading"+"_"+model].astype(float) > 0.875
        df["MPI"+"_"+model] = df["distributed memory"+"_"+model].astype(float) > 0.875
        df["2D"+"_"+model] = df["two dimensional"+"_"+model].astype(float) > 0.875
        df["1D"+"_"+model] = df["one dimensional"+"_"+model].astype(float) > 0.875
        df["DET"+"_"+model] = df["deterministic"+"_"+model].astype(float) > 0.875
        df["STC"+"_"+model] = df["stochastic"+"_"+model].astype(float) > 0.875
        df["FVM"+"_"+model] = df["finite volume method"+"_"+model].astype(float) > 0.875
        df["FEM"+"_"+model] = df["finite element method"+"_"+model].astype(float) > 0.875
        df["FDM"+"_"+model] = df["finite difference method"+"_"+model].astype(float) > 0.875
        df["NMD"+"_"+model] = df["numerical method development"+"_"+model].astype(float) > 0.875
        df["RBM"+"_"+model] = df["riverbed morphology"+"_"+model].astype(float) > 0.875
        df["GHM"+"_"+model] = df["hydrological modeling"+"_"+model].astype(float) > 0.875
        df["OM"+"_"+model] = df["ocean modeling"+"_"+model].astype(float) > 0.875
        df["CPR"+"_"+model] = df["computational performance report"+"_"+model].astype(float) > 0.875
        df["VNM"+"_"+model] = df["validation of numerical method"+"_"+model].astype(float) > 0.875
        df["IBER"+"_"+model] = df["iber"+"_"+model].astype(float) > 0.875
        df["TELEMAC"+"_"+model] = df["telemac"+"_"+model].astype(float) > 0.875
        df["HEC-RAS"+"_"+model] = df["hec-ras"+"_"+model].astype(float) > 0.875

        fig, ax = plt.subplots(figsize=(14,6))

        trues = []
        values = None

        for label in labels:
            classes, values = np.unique(df[label+"_"+model], return_counts=True)
            trues.append(100*(len(df[label+"_"+model])-values[0])/len(df[label+"_"+model]))

        z = [x for _, x in sorted(zip(trues, labels), reverse=True)]

        #plt.bar(labels, falses, color=custom_colors[0])
        plt.figure(figsize=(16,4))
        plt.bar(z, sorted(trues, reverse=True), color=custom_colors[1])
        for i in range(len(labels)):
            plt.text(z[i], sorted(trues, reverse=True)[i]/2, "{:02.1f}".format(sorted(trues, reverse=True)[i]), ha="center")
        plt.grid()
        plt.xlabel("Topic Abbreviation")
        plt.ylabel("Percentage of papers")
        plt.text(0.9, 0.9, "Total papers: "+str(np.sum(values)), size=10, transform=plt.gca().transAxes,
                ha="center", va="center",
                bbox=dict(boxstyle="round",
                        ec=(0., 0.5, 0.5),
                        fc=(1., 1, 1),
                        )
                )
        plt.savefig("overall_"+model+".png")
        plt.close()

    return df

def get_overall_classes(df, models, labels):
    
    trues = []
    values = None
    for label in labels:
        # Get classes of both models
        # Select columns for the current label
        cols = ["{}_{}".format(label, model) for model in models]
        
        df[label] = df[cols].any(axis=1)
        #df[df[label]]["Title"].to_csv(label+".csv")

        classes, values = np.unique(df[label], return_counts=True)
        trues.append(100*(len(df[label])-values[0])/len(df[label]))

    z = [x for _, x in sorted(zip(trues, labels), reverse=True)]

    plt.figure(figsize=(16,4))
    plt.bar(z, sorted(trues, reverse=True), color=custom_colors[1])
    for i in range(len(labels)):
        plt.text(z[i], sorted(trues, reverse=True)[i]/2, "{:02.1f}".format(sorted(trues, reverse=True)[i]), ha="center")
    plt.grid()
    plt.xlabel("Topic Abbreviation")
    plt.ylabel("Percentage of papers")
    plt.text(0.9, 0.9, "Total papers: "+str(np.sum(values)), size=10, transform=plt.gca().transAxes,
            ha="center", va="center",
            bbox=dict(boxstyle="round",
                    ec=(0., 0.5, 0.5),
                    fc=(1., 1, 1),
                    )
            )

    plt.savefig("overall.png")
    plt.close()
    
    return df

def filter_classes(df):
    #considering all which regard numerical method development
    df_NMD = df[df["NMD"]]
    # we only want the ones that have to dow with 2 dimensional modeling
    df_2D = df_NMD[df_NMD["2D"]]
    # We only consider the ones that are not explicitly stochastic
    df_2D = df_2D[np.logical_not(df_2D["STC"])]
    # We will classify then the ones that are related to HPC and the ones that are not 
    print(np.unique(df_2D["HPC"], return_counts=True))
    
    df_2D[["Title", "Abstract Note"]].to_csv("2D_read.csv")
    df_2D[np.logical_not(df_2D["HPC"])][["Title", "Abstract Note"]].to_csv("notHPC.csv")
    df_2D[df_2D["HPC"]][["Title", "Abstract Note"]].to_csv("2DHPC.csv")


if __name__ == "__main__":

    df = pd.read_csv("classified.csv")
    graph_sources(df)
    graph_distribution(df)

    models = [
        "bart-large-mnli", 
        "deberta-v3-large-zeroshot-v2.0"
    ]

    labels = ["BLT", "MB", "HPC", "GPU", "CPU", "MPI", "2D", "1D", "DET", "STC", "FVM", "FEM", "FDM", "NMD", "RBM", "GHM", "OM", "CPR", "VNM", "IBER", "TELEMAC", "HEC-RAS"]

    df = plot_class_distribution(df, models, labels)
    df = get_overall_classes(df, models, labels)
    df = filter_classes(df)