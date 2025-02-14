import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load training data from train.txt
train_data = []
train_labels = []
with open("train_data.txt", "r", encoding="utf-8") as file:
    content = file.read().strip().split(":::\n")  # Assuming ':::' separates entries
    for entry in content:
        parts = entry.strip().split(":::")
        if len(parts) == 2:
            genre, plot = parts
            train_data.append(plot)
            train_labels.append(genre)
