import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('data_n.csv')

filtered_df = df[(df["account_balance_13_march"] > 0) & (df["account_balance_18_march"] < 0)]

resolved_counts = filtered_df["resolved"].value_counts()

# Extract counts for "resolved" and "floor"
count_resolved = resolved_counts.get("resolved", 0)  # Default to 0 if "resolved" is not found
count_floor = resolved_counts.get("floor", 0)  # Default to 0 if "floor" is not found
print(f"Percentage of 'resolved': {count_resolved * 100/ (count_resolved+count_floor)}%")
print(f"Percentage of 'floor': {count_floor * 100/ (count_resolved+count_floor)}%")
print(f"total count: {count_resolved+count_floor}")

# Observation: out of total dataset, floored call is around 20%. but with 13Balance >0 & 18Balance < 0, then floor shoots to 42.5%