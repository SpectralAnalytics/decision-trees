import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_iris

st.set_page_config(layout="wide")

st.title("How a Decision Tree Works")

st.subheader("A Demonstration by Marcel")

st.write("""
The Iris dataset is well known for testing out machine learning classification algorithms. Here is a pair plot of the four features colored by species.
""")

df = sns.load_dataset("iris")

st.pyplot(sns.pairplot(df, hue="species"))

st.write("""
Imagine that you can only use one feature at any one time to compare and classify data. Which feature would you pick first and what value would you pick to perform the split?
""")

st.write("""Hint: You can pick "petal length (cm)" as the first feature with a threshold of 2.45""")

col1, col2 = st.columns(2)

with col1:
    feature1 = st.selectbox(
        "First feature to split by:",
        df.columns[:-1],
        index=None,
        placeholder="Please select the feature...",
    )
with col2:
    if feature1 is not None:
        feature = df[feature1]
        value1 = st.slider(
            "Value of first feature to split by:",
            feature.min(), feature.max(),
        )
    else:
        value1 = st.slider(
            "Value of first feature to split by:",
            0, 1,
            disabled=True
        )

if value1 == 0:
    st.write("An exhibit will appear once you have set the feature and threshold correctly.")
else:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("Histogram by Species")
        plt.figure() # clear the previous 
        sns.kdeplot(data=df, x=feature1, hue="species")
        plt.axvline(x=value1, color="red", label="Threshold")
        st.pyplot(plt)

    with col2:
        st.header("Left Node Analysis")
        left_node = df[df[feature1]<=value1]
        target = left_node["species"]
        left_counts = target.value_counts()
        left_count = left_counts.sum()
        markdown = " - ".join(left_counts.map(lambda x: f"\\left(\\frac{{{x}}}{{{len(target)}}}\\right)^{{2}}"))
        left_gini = 1 - np.power(left_counts/len(target), 2).sum()
        st.markdown(f"$\\mathrm{{Gini}} = 1 - {markdown} = {left_gini:0.3f}$")
        st.write(left_node)

    with col3:
        st.header("Right Node Analysis")
        right_node = df[df[feature1]>value1]
        target = right_node["species"]
        right_counts = target.value_counts()
        right_count = right_counts.sum()
        markdown = " - ".join(right_counts.map(lambda x: f"\\left(\\frac{{{x}}}{{{len(target)}}}\\right)^{{2}}"))
        right_gini = 1 - np.power(right_counts/len(target), 2).sum()
        st.markdown(f"$\\mathrm{{Gini}} = 1 - {markdown} = {right_gini:0.3f}$")
        st.write(right_node)

    gini = (left_gini*left_count + right_gini*right_count)/(left_count + right_count)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"$\\mathrm{{Overall\\ Gini}} = \\frac{{{left_gini:0.6f} \\times {left_count} + {right_gini:0.6f} \\times {right_count}}}{{ {left_count} + {right_count}}} = {gini:0.6f}$")

    with col2:
        st.link_button("Read more on Wikipedia", "https://en.wikipedia.org/wiki/Decision_tree_learning")
