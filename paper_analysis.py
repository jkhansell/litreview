import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import re
plt.rcParams['axes.axisbelow'] = True

import gensim.downloader
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

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
    return re.sub(r'\W+', '', text.lower().strip())

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

def ngram_tokenizer(n, cleaned_docs):
    ngramed_docs = cleaned_docs

    for _ in range(n):
        ngram = Phrases(ngramed_docs, min_count=20, threshold=10, delimiter=" ")
        ngram_phraser = Phraser(ngram)

        ngramed_tokens = []
        for sent in ngramed_docs:
            tokens = ngram_phraser[sent]
            ngramed_tokens.append(tokens)
        
        ngramed_docs = ngramed_tokens
    return ngramed_tokens

def train_glove_titles(glove_model, df): 
    documents = df["gensim_tokenized_abstracts"]
    titles = [title for title in documents]

    cleaned = []
    for title in titles:
        cleaned_title = [word.lower() for word in title]
        cleaned_title = [word for word in title if word not in stopwords]
        cleaned.append(cleaned_title)
    
    ngram_tokens = ngram_tokenizer(3, cleaned)

    # build a toy model to update with
    base_model = Word2Vec(vector_size=300, min_count=5)
    base_model.build_vocab(ngram_tokens)
    # add GloVe's vocabulary & weights
    base_model.build_vocab([glove_model.index_to_key], update=True)
    total_examples = base_model.corpus_count

    # train on our data
    base_model.train(ngram_tokens, total_examples=total_examples, epochs=10)
    return base_model

def test_w2v(model, words):
    for word in words:
        math_result = model.most_similar(word, topn=5)
        print(f'Word: - {word}')
        [print(f"- {result[0]} ({round(result[1],5)})") for result in math_result]
        print()

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

if __name__ == "__main__":
    sources = ["Google-Scholar", "Web-of-Science", "IEEE", "Scopus"]
    df = unify_databases(sources)
    #graph_sources(df, name="_raw")
    #graph_distribution(df, name="_raw")
    # clean data
    df = clean_dataframe(df)
    #graph_distribution(df, name="_processed")
    #graph_sources(df, name="_processed")
    # load model
    #df = gensim_tokenize_data(df)
    #glove_model = KeyedVectors.load_word2vec_format('/work/jovillalobos/glove.840B.300d.bin', binary=True)
    #glove_model = KeyedVectors.load_word2vec_format('/work/jovillalobos/glove.840B.300d.txt', binary=False, no_header=True)
    glove_model = KeyedVectors.load("/work/jovillalobos/glove.840B.300d.bin", mmap="r")
    #model = train_glove_titles(glove_model, df)
    words = ["shallow", "water", "equations"]
    test_w2v(glove_model, words)
