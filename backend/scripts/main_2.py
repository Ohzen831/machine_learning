# Cell type: code
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.metrics import accuracy_score, precision_score

from category_encoders import BinaryEncoder

from imblearn.over_sampling import SMOTE

from joblib import dump, load

import graphviz

# Cell type: code
# Read the Netflix dataset

netflix_data = pd.read_csv('../data/netflix_titles.csv')



# Cell 3: Display basic info

netflix_data.info()

netflix_data.head()

# Cell type: code
# drop rows with missing critical values like type, title, release_year, duration, listed_in

cleaned_data = netflix_data.dropna(subset=["type", "title", "release_year", "duration", "listed_in", "country"])

# Cell type: code
#Remove duplicate rows based on specified columns

cleaned_data = cleaned_data.drop_duplicates(subset=["title"])

# Cell type: code
print(cleaned_data['type'].unique())

# Cell type: code
# Cell 6: Visual Exploration

import matplotlib.pyplot as plt

import seaborn as sns



# Set visual style

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))



# 1. Count of Movies vs TV Shows

plt.subplot(2, 2, 1)

sns.countplot(data=cleaned_data, x="type", palette="Set2")

plt.title("Count of Movies vs TV Shows")

plt.xlabel("Type")

plt.ylabel("Count")



# 2. Top 10 Countries Producing Content

top_countries = cleaned_data['country'].value_counts().head(10)

plt.subplot(2, 2, 2)

sns.barplot(y=top_countries.index, x=top_countries.values, palette="Set3")

plt.title("Top 10 Countries Producing Content")

plt.xlabel("Number of Titles")

plt.ylabel("Country")


# Cell type: code
# 3. Count of Titles per Release Year (last 20 years)

recent_years = cleaned_data[cleaned_data['release_year'] >= cleaned_data['release_year'].max() - 20]

year_counts = recent_years['release_year'].value_counts().sort_index()

plt.subplot(2, 2, 3)

sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o', color="steelblue")

plt.title("Titles Released per Year (Recent 20 Years)")

plt.xlabel("Release Year")

plt.ylabel("Number of Titles")



# 4. Most Frequent Genres (Top 10)

from collections import Counter

# Flatten genre list

genre_list = cleaned_data['listed_in'].str.split(', ').sum()

top_genres = Counter(genre_list).most_common(10)

genres, counts = zip(*top_genres)

plt.subplot(2, 2, 4)

sns.barplot(x=list(counts), y=list(genres), palette="viridis")

plt.title("Top 10 Most Frequent Genres")

plt.xlabel("Count")

plt.ylabel("Genre")



plt.tight_layout()

plt.show()


# Cell type: code
#Feature Engineering & Model Preparation

# Cell type: code
# Clone the cleaned dataset to avoid modifying original

feature_data = cleaned_data.copy()

# Cell type: code
from sklearn.preprocessing import LabelEncoder



# Initialize the LabelEncoder

type_encoder = LabelEncoder()



# Apply encoding to the 'type' column

feature_data['type_encoded'] = type_encoder.fit_transform(feature_data['type'])

# Check the encoded values



print(feature_data['type_encoded'].head())

print(feature_data['type_encoded'].unique())

# Cell type: code
feature_data['type'] = feature_data['type'].map({'Movie': 0, 'TV Show': 1})

# Cell type: code
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

import numpy as np



# 2. Extract numerical duration

# Convert 'duration' column to numeric: extract minutes or seasons

def parse_duration(x):

    try:

        return int(x.split(' ')[0])

    except:

        return np.nan



feature_data['parsed_duration'] = feature_data['duration'].apply(parse_duration)

# Cell type: code
# 3. Encode genres using MultiLabelBinarizer

# 'listed_in' contains comma-separated genres -> we turn them into binary features

mlb = MultiLabelBinarizer()

genres_encoded = mlb.fit_transform(feature_data['listed_in'].str.split(', '))

genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

# Cell type: code
# 4. Encode country using Label Encoding (for simplicity — OneHot can be too sparse)

from sklearn.preprocessing import LabelEncoder

country_encoder = LabelEncoder()

feature_data['country_encoded'] = country_encoder.fit_transform(feature_data['country'])

# Cell type: code
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import NearestNeighbors



def recommend_content(genre_input, type_input, release_year_input, country_input):

    """

    Recommend Netflix content matching:

    - At least 2 selected genres

    - Content type (Movie/TV Show)

    - Country (case-insensitive partial match)

    - Release year >= selected year

    """

    # Normalize genre input

    genre_input = [g.strip().title() for g in genre_input]



    # Step 1: Filter data by type, country, and release year range

    filtered_data = cleaned_data[

        (cleaned_data['type'] == ('Movie' if type_input == 0 else 'TV Show')) &

        (cleaned_data['country'].str.contains(country_input, case=False, na=False)) &

        (cleaned_data['release_year'] >= release_year_input)

    ]



    # Step 2: Match at least 2 selected genres

    def genres_match_at_least_two(genres_str):

        content_genres = [g.strip() for g in genres_str.split(',')]

        return sum(1 for g in genre_input if g in content_genres) >= 2



    filtered_data = filtered_data[filtered_data['listed_in'].apply(genres_match_at_least_two)]



    if filtered_data.empty:

        print("No content found matching your criteria.")

        return pd.DataFrame()



    # Step 3: Prepare features for NearestNeighbors

    filtered_indices = filtered_data.index.intersection(genres_df.index)

    filtered_genres_df = genres_df.loc[filtered_indices]

    filtered_features = pd.concat([

        filtered_data[['type', 'release_year', 'parsed_duration', 'country_encoded']],

        filtered_genres_df

    ], axis=1).fillna(0)



    # Step 4: Fit Nearest Neighbors model on filtered data

    filtered_scaled = scaler.transform(filtered_features)

    local_model = NearestNeighbors(n_neighbors=min(10, len(filtered_data)), algorithm='auto')

    local_model.fit(filtered_scaled)



    # Step 5: Create scaled user input vector

    avg_duration = feature_data[feature_data['type'] == type_input]['parsed_duration'].mean()

    user_input = pd.DataFrame([{

        'type': type_input,

        'release_year': release_year_input,

        'parsed_duration': avg_duration,

        'country_encoded': country_encoder.transform([country_input])[0]

    }])



    for genre in mlb.classes_:

        user_input[genre] = 1 if genre in genre_input else 0



    user_input_scaled = scaler.transform(user_input)



    # Step 6: Get nearest neighbors

    distances, indices = local_model.kneighbors(user_input_scaled)

    recommendations = filtered_data.iloc[indices[0]]



    return recommendations[['title', 'listed_in', 'type', 'duration', 'country', 'rating', 'description']]


# Cell type: code
# 5. Final feature set

from sklearn.neighbors import NearestNeighbors



final_features = pd.concat([

    feature_data[['type', 'release_year', 'parsed_duration', 'country_encoded']],

    genres_df

], axis=1)



# Handle missing values (e.g., NaN in parsed_duration)

final_features = final_features.fillna(0)



# Normalize features for Nearest Neighbors

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(final_features)



# --- Train Nearest Neighbors Model ---

model = NearestNeighbors(n_neighbors=10, algorithm='auto')

model.fit(X_scaled)



print("Feature engineering complete and Nearest Neighbors model is trained.")

# Cell type: code
#User Input System & Recommendation Generation

# Cell type: code
def recommend_content(genre_input, type_input, release_year_input, country_input):

    # Normalize genre input

    genre_input = [g.strip().title() for g in genre_input]



    # Use average duration for the selected type

    avg_duration = feature_data[feature_data['type'] == type_input]['parsed_duration'].mean()



    # Step 1: Create user input features

    user_input = pd.DataFrame([{

        'type': type_input,

        'release_year': release_year_input,

        'parsed_duration': avg_duration,

        'country_encoded': country_encoder.transform([country_input])[0]

    }])



    # Step 2: Add genre columns

    for genre in mlb.classes_:

        user_input[genre] = 0

    for genre in genre_input:

        if genre in mlb.classes_:

            user_input[genre] = 1



    # Step 3: Scale the input

    user_input_scaled = scaler.transform(user_input)



    # Step 4: Find nearest neighbors

    distances, indices = model.kneighbors(user_input_scaled)



    # Step 5: Get recommendations

    recommendations = cleaned_data.iloc[indices[0]]

    return recommendations[['title', 'listed_in', 'type', 'duration', 'country', 'rating', 'description']]

# Cell type: code
# Example: Get user preferences

# For testing, you can provide the following sample inputs (you can later collect inputs interactively):

genre_input = ['Action', 'Drama']  # User prefers these genres

type_input = 0  # User selected "Movie"

release_year_input = 2021  # User prefers movies released in 2021

country_input = 'United States'  # User prefers US-based content



# Call the updated recommend_content function (no duration_input)

recommendations = recommend_content(genre_input, type_input, release_year_input, country_input)



# Display the recommended titles

print(recommendations)


# Cell type: code
#DECISION TREE: Model Training & Evaluation for Rating predicion

# Cell type: code
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import (accuracy_score, precision_score, classification_report, confusion_matrix)

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



# 1. Enhanced Rating Cleaning and Preparation

def prepare_ratings(y):

    """Clean and standardize ratings with comprehensive mapping"""

    rating_map = {

        'G': 'G',

        'PG': 'PG',

        'PG-13': 'PG-13',

        'R': 'R',

        'NC-17': 'NC-17',

        'TV-Y': 'TV-Y',

        'TV-Y7': 'TV-Y7',

        'TV-Y7-FV': 'TV-Y7-FV',

        'TV-G': 'TV-G',

        'TV-PG': 'TV-PG',

        'TV-14': 'TV-14',

        'TV-MA': 'TV-MA'

    }

    

    # Filter out durations and unknown ratings

    y = y[~y.astype(str).str.contains('min')]  # Remove duration entries

    y = y.map(rating_map).dropna()  # Apply mapping and drop NA/unmapped

    

    return y



# Prepare target

y = prepare_ratings(feature_data['rating'])

X = final_features.loc[y.index]  # Align features

# Cell type: code
# 2. Improved Train-Test Split with Stratification

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42, stratify=y)

# Cell type: code
# 3. Enhanced Model Selection and Training

# Using RandomForest for better performance

clf = RandomForestClassifier(

    n_estimators=200,

    max_depth=10,

    min_samples_split=5,

    class_weight='balanced',  # Handle class imbalance

    random_state=42

)



# Cross-validation

cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

print(f"Cross-validated Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")



# Train model

clf.fit(X_train, y_train)

# Cell type: code
# 4. Comprehensive Evaluation

y_pred = clf.predict(X_test)



print("\n=== Detailed Classification Report ===")

print(classification_report(y_test, y_pred, zero_division=0))



print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}")

print(f"Weighted Precision: {precision_score(y_test, y_pred, average='weighted') * 100:.2f}")



# Confusion Matrix Visualization

plt.figure(figsize=(12, 8))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)

plt.title('Rating Prediction Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()

# Cell type: code
# 5. Feature Importance Analysis

feature_imp = pd.DataFrame({

    'feature': X.columns,

    'importance': clf.feature_importances_

}).sort_values('importance', ascending=False)



plt.figure(figsize=(10, 6))

sns.barplot(x='importance', y='feature', data=feature_imp.head(15))

plt.title('Top 15 Important Features for Rating Prediction')

plt.show()

# Cell type: code
# 6. Production-Ready Prediction Function

def predict_rating(features):

    """Predict rating with confidence scores"""

    pred = clf.predict(features)

    proba = clf.predict_proba(features)

    

    results = []

    for i, rating in enumerate(pred):

        confidence = proba[i][clf.classes_ == rating][0]

        results.append({

            'predicted_rating': rating,

            'confidence': round(confidence, 2),

            'alternative_ratings': dict(zip(

                clf.classes_,

                [round(p, 2) for p in proba[i]]

            ))

        })

    

    return results

# Cell type: code
from sklearn.tree import DecisionTreeClassifier, export_graphviz



# 1. Train a visualization-friendly Decision Tree

tree_model = DecisionTreeClassifier(

    max_depth=4,               # Limited for interpretability

    min_samples_split=20,      # Prevent overfitting

    class_weight='balanced',   # Handle imbalanced classes

    random_state=42

)

tree_model.fit(X_train, y_train)



# 2. Export the decision tree to DOT file

export_graphviz(

    tree_model,

    out_file='../outputs/rating_tree.dot',  # Output file path

    feature_names=X.columns,                # Feature names

    class_names=tree_model.classes_,        # Rating categories

    filled=True,                            # Color coding

    rounded=True,                           # Rounded boxes

    special_characters=True                 # Special symbols

)



print("Decision tree exported to rating_tree.dot")

# Cell type: code
import joblib



# Save the model with joblib

joblib.dump(clf, '../data/rating_predictor.joblib')



# Verify the classes are what we expect

print("Model saved with these rating classes:")

print(clf.classes_)

