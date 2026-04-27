import numpy as np
import joblib
import os
import re
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from scipy.stats import randint, uniform

from data_loader import load_data


_ROMAN = {'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7,'VIII':8,'IX':9,
          'X':10,'XI':11,'XII':12,'XIII':13,'XIV':14,'XV':15,'XVI':16,
          'XVII':17,'XVIII':18,'XIX':19,'XX':20,'XXI':21,'XXII':22,'XXIII':23}

def roman_to_int(name):
    m = re.match(r'^([IVXL]+)\.', str(name))
    return _ROMAN.get(m.group(1), 0) if m else 0

#Adat betöltés és szűrés

df = load_data("../listings.csv")

q1, q3 = df['price'].quantile(0.25), df['price'].quantile(0.75)
iqr = q3 - q1
low, high = max(q1 - 1.5 * iqr, 1), q3 + 1.5 * iqr
df = df[(df['price'] >= low) & (df['price'] <= high)]
print(f"Sorok: {len(df)} | Ár-tartomány: {low:.0f}–{high:.0f} Ft/éj")

# Feature engineering

df['district_num']  = df['neighbourhood_cleansed'].apply(roman_to_int).astype(float)
df['review_weight'] = np.log1p(df['number_of_reviews']) * df['review_scores_rating']
df['availability_ratio'] = df['availability_30'] / 30.0


FEATURES_NUM = [
    'calculated_host_listings_count',
    'bathrooms',
    'bedrooms', 'longitude', 'latitude',
    'review_scores_rating', 'minimum_nights',
    'review_scores_cleanliness', 'review_scores_location',
    'availability_30',
    'district_num',
    'host_response_rate', 'review_weight',
    'number_of_reviews',
]

FEATURES_CAT = ['room_type', 'neighbourhood_cleansed', 'property_type_grouped']

X = df[FEATURES_NUM + FEATURES_CAT]
y = np.log1p(df['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_test_orig = np.expm1(y_test).values

print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(FEATURES_NUM) + len(FEATURES_CAT)}")

# Pipeline + hiperparaméter keresés

preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', FEATURES_NUM),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), FEATURES_CAT),
])

pipe = Pipeline([
    ('prep', preprocessor),
    ('model', HistGradientBoostingRegressor(
        random_state=42, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20,
    )),
])

param_dist = {
    'model__max_iter':          randint(800, 2500),
    'model__max_depth':         randint(5, 15),
    'model__learning_rate':     uniform(0.01, 0.12),
    'model__min_samples_leaf':  randint(5, 40),
    'model__max_leaf_nodes':    randint(31, 511),
    'model__l2_regularization': uniform(0, 3.0),
}

print("\nHiperparaméter keresés ")

search = RandomizedSearchCV(
    pipe, param_distributions=param_dist,
    n_iter=100, cv=5, scoring='neg_mean_absolute_error',
    random_state=42, n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

best = search.best_estimator_
print(f"\nLegjobb paraméterek:")
for k, v in search.best_params_.items():
    print(f"   {k}: {v}")

# Kiértékelés

preds = np.maximum(np.expm1(best.predict(X_test)), 0)
mae  = mean_absolute_error(y_test_orig, preds)
rmse = root_mean_squared_error(y_test_orig, preds)
r2   = r2_score(y_test_orig, preds)
mask = y_test_orig > 0
mape = np.mean(np.abs((y_test_orig[mask] - preds[mask]) / y_test_orig[mask])) * 100

print(f"\n{'═'*50}")
print(f"  MAE  : {mae:>10,.0f} Ft/éj")
print(f"  RMSE : {rmse:>10,.0f} Ft/éj")
print(f"  R²   : {r2:>10.4f}")
print(f"  MAPE : {mape:>10.1f} %")
print(f"{'═'*50}")

#  Mentés

os.makedirs("model", exist_ok=True)
joblib.dump(best, "model/price_model.pkl")
joblib.dump(FEATURES_NUM + FEATURES_CAT, "model/feature_names.pkl")
joblib.dump(True, "model/log_target.pkl")
print("\nModell mentve: model/price_model.pkl")

# Kerületi statisztikák
district_stats = (
    df.groupby('neighbourhood_cleansed')
    .agg(avg_price=('price', 'mean'), median_price=('price', 'median'),
         count=('price', 'count'), avg_lat=('latitude', 'mean'), avg_lon=('longitude', 'mean'))
    .reset_index()
)
district_stats.to_csv("model/district_stats.csv", index=False)
print("Kerületi statisztikák: model/district_stats.csv")