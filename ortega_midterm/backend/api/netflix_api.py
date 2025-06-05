# netflix_api.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import os
import joblib
import warnings
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost"}})

# Load the preprocessed data and models
try:
    # Load the cleaned data - Using absolute path approach from working code
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, '..', 'data', 'netflix_titles.csv')
    cleaned_data = pd.read_csv(csv_path)

    cleaned_data = cleaned_data.dropna(subset=["type", "title", "release_year", "duration", "listed_in", "country"])
    cleaned_data = cleaned_data.drop_duplicates(subset=["title"])
    
    # Prepare feature data (replicating the preprocessing steps)
    feature_data = cleaned_data.copy()
    feature_data['type'] = feature_data['type'].map({'Movie': 0, 'TV Show': 1})
    
    # Duration parsing
    def parse_duration(x):
        try:
            return int(x.split(' ')[0])
        except:
            return np.nan
    feature_data['parsed_duration'] = feature_data['duration'].apply(parse_duration)
    
    # Genre encoding
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(feature_data['listed_in'].str.split(', '))
    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
    
    # Country encoding
    country_encoder = LabelEncoder()
    feature_data['country_encoded'] = country_encoder.fit_transform(feature_data['country'])
    
    # Final features
    final_features = pd.concat([
        feature_data[['type', 'release_year', 'parsed_duration', 'country_encoded']],
        genres_df
    ], axis=1).fillna(0)
    
    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(final_features)
    
    # Load or train the recommendation model
    recommendation_model = NearestNeighbors(n_neighbors=10, algorithm='auto')
    recommendation_model.fit(X_scaled)
    
    # Load the rating prediction model - Note: this will be skipped if file doesn't exist
    rating_model_path = os.path.join(base_dir, '..', 'data', 'rating_predictor.joblib')
    try:
        rating_model = joblib.load(rating_model_path)
        has_rating_model = True
    except:
        print("Rating model not found. Rating prediction will be unavailable.")
        has_rating_model = False
    
    print("Models and data loaded successfully!")

except Exception as e:
    print(f"Error loading models or data: {str(e)}")
    raise e


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({"status": "healthy", "message": "Netflix recommendation API is running"})


@app.route('/api/genres', methods=['GET'])
def get_genres():
    """Return all available genres in the dataset."""
    return jsonify({"genres": sorted(list(mlb.classes_))})


@app.route('/api/countries', methods=['GET'])
def get_countries():
    """Return all available countries in the dataset."""
    countries = sorted(cleaned_data['country'].dropna().unique().tolist())
    return jsonify({"countries": countries})


@app.route('/recommend', methods=['POST'])
def recommend():
    """Endpoint for content recommendations based on user preferences"""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['genres', 'content_type', 'release_year', 'country']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Process input
        genre_input = data['genres']
        type_input = 0 if data['content_type'].lower() == 'movie' else 1
        release_year_input = int(data['release_year'])
        country_input = data['country']
        
        # Use average duration for the selected type
        duration_input = feature_data[feature_data['type'] == type_input]['parsed_duration'].mean()
        
        # Create feature array from user input
        user_input = pd.DataFrame([{
            'type': type_input,
            'release_year': release_year_input,
            'parsed_duration': duration_input,
            'country_encoded': country_encoder.transform([country_input])[0]
        }])
        
        # Add binary genre columns
        for genre in mlb.classes_:
            user_input[genre] = 0
        for genre in genre_input:
            if genre in mlb.classes_:
                user_input[genre] = 1
        
        # Normalize and get recommendations
        user_input_scaled = scaler.transform(user_input)
        distances, indices = recommendation_model.kneighbors(user_input_scaled)
        recommendations = cleaned_data.iloc[indices[0]]
        
        # Format response
        results = []
        for _, row in recommendations.iterrows():
            results.append({
                'title': row['title'],
                'type': row['type'],
                'genres': row['listed_in'],
                'duration': row['duration'],
                'country': row['country'],
                'rating': row['rating'] if 'rating' in row and pd.notna(row['rating']) else None,
                'description': row['description'] if 'description' in row and pd.notna(row['description']) else None,
                'release_year': row['release_year']
            })
        
        return jsonify({'recommendations': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommend', methods=['POST'])
def recommend_api():
    """New API endpoint for content recommendations"""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['genres', 'type', 'release_year', 'country']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Process input
        genre_input = data['genres']
        type_input = data['type']  # Expecting 0 for Movie, 1 for TV Show
        release_year_input = int(data['release_year'])
        country_input = data['country']
        
        # Use average duration for the selected type
        duration_input = feature_data[feature_data['type'] == type_input]['parsed_duration'].mean()
        
        # Create feature array from user input
        user_input = pd.DataFrame([{
            'type': type_input,
            'release_year': release_year_input,
            'parsed_duration': duration_input,
            'country_encoded': country_encoder.transform([country_input])[0]
        }])
        
        # Add binary genre columns
        for genre in mlb.classes_:
            user_input[genre] = 0
        for genre in genre_input:
            if genre in mlb.classes_:
                user_input[genre] = 1
        
        # Normalize and get recommendations
        user_input_scaled = scaler.transform(user_input)
        distances, indices = recommendation_model.kneighbors(user_input_scaled)
        recommendations = cleaned_data.iloc[indices[0]]
        
        # Format response
        results = []
        for _, row in recommendations.iterrows():
            results.append({
                'title': row['title'],
                'type': row['type'],
                'genres': row['listed_in'],
                'duration': row['duration'],
                'country': row['country'],
                'rating': row['rating'] if 'rating' in row and pd.notna(row['rating']) else None,
                'description': row['description'] if 'description' in row and pd.notna(row['description']) else None,
                'release_year': row['release_year']
            })
        
        return jsonify({
            "recommendations": results,
            "count": len(results),
            "parameters": {
                "genres": genre_input,
                "type": "Movie" if type_input == 0 else "TV Show",
                "release_year": release_year_input,
                "country": country_input
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_rating', methods=['POST'])
def predict_rating():
    """Endpoint for predicting content rating based on features"""
    if not has_rating_model:
        return jsonify({'error': 'Rating prediction model not available'}), 503
        
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['content_type', 'release_year', 'country', 'genres']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Process input
        type_input = 0 if data['content_type'].lower() == 'movie' else 1
        release_year_input = int(data['release_year'])
        country_input = data['country']
        genre_input = data['genres']
        
        # Use average duration for the selected type
        duration_input = feature_data[feature_data['type'] == type_input]['parsed_duration'].mean()
        
        # Create feature array
        features = pd.DataFrame([{
            'type': type_input,
            'release_year': release_year_input,
            'parsed_duration': duration_input,
            'country_encoded': country_encoder.transform([country_input])[0]
        }])
        
        # Add genre columns
        for genre in mlb.classes_:
            features[genre] = 0
        for genre in genre_input:
            if genre in mlb.classes_:
                features[genre] = 1
        
        # Make prediction
        pred = rating_model.predict(features)
        proba = rating_model.predict_proba(features)
        
        # Format response
        result = {
            'predicted_rating': pred[0],
            'confidence': round(float(proba[0][np.where(rating_model.classes_ == pred[0])[0][0]]), 2),
            'possible_ratings': dict(zip(
                rating_model.classes_.tolist(),
                [round(float(p), 2) for p in proba[0]]
            ))
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/available_genres', methods=['GET'])
def available_genres():
    """Endpoint to get all available genres"""
    return jsonify({'genres': mlb.classes_.tolist()})


@app.route('/available_countries', methods=['GET'])
def available_countries():
    """Endpoint to get all available countries"""
    countries = cleaned_data['country'].dropna().unique().tolist()
    return jsonify({'countries': countries})


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)