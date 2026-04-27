import pandas as pd
import numpy as np

def load_data(path="../listings.csv"):
    cols = [
        "id", "longitude", "latitude",
        "neighbourhood_cleansed",
        "room_type", "property_type",
        "bedrooms", "beds", "bathrooms",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "review_scores_rating", "review_scores_cleanliness",
        "review_scores_location",
        "host_is_superhost", "instant_bookable",
        "availability_30",
        "calculated_host_listings_count", "host_response_rate",
    ]

    df = pd.read_csv(path, low_memory=False)[cols].copy()

    # Ár konvertálás
    df['price'] = df['price'].astype(str).str.replace('[$,]', '', regex=True).astype(float)
    df = df.dropna(subset=['price'])

    # Százalékos mező
    df['host_response_rate'] = df['host_response_rate'].astype(str).str.replace('%', '', regex=False)
    df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce')
    df['host_response_rate'] = df['host_response_rate'].fillna(df['host_response_rate'].median())

    # Hiányzó értékek kitöltése
    median_cols = ['bedrooms', 'beds', 'bathrooms',
                   'review_scores_cleanliness', 'review_scores_location',
                   'availability_30']
    df[median_cols] = df[median_cols].fillna(df[median_cols].median())


    df['calculated_host_listings_count'] = df['calculated_host_listings_count'].fillna(1)
    df['property_type'] = df['property_type'].fillna('Other')

    # Értékelés
    df['has_rating'] = df['review_scores_rating'].notna().astype(int)
    df['review_scores_rating'] = df['review_scores_rating'].fillna(df['review_scores_rating'].median())

    # Bináris konverzió
    for col in ['host_is_superhost', 'instant_bookable']:
        df[col] = (df[col] == 't').astype(int)



    # Property type csoportosítás
    top_types = df['property_type'].value_counts()
    rare_types = top_types[top_types < 30].index
    df['property_type_grouped'] = df['property_type'].replace(rare_types, 'Other')

    return df