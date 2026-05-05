import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

class NumericModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.model_path = 'models/numeric_rf.joblib'
        self.scaler_path = 'models/scaler.joblib'

    def train(self, data_path='data/properties.csv'):
        df = pd.DataFrame()
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            raise FileNotFoundError(f"Data not found at {data_path}")

        # Advanced Feature Engineering
        df['price_per_sqm'] = df['price'] / df['size']
        df['rooms_per_sqm'] = df['rooms'] / df['size']
        df['is_studio'] = (df['rooms'] == 1).astype(int)
        
        # Features for the model
        X = df[['size', 'rooms', 'rooms_per_sqm', 'is_studio']]
        y = df['price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        # Ultimate RF Configuration
        self.model = RandomForestRegressor(
            n_estimators=500, 
            max_depth=15,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Save models
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        score = self.model.score(self.scaler.transform(X_test), y_test)
        print(f"Numerical Model Trained. R2 Score: {score:.4f}")

    def predict_fair_price(self, size, rooms):
        if not hasattr(self, 'loaded_model'):
            self.loaded_model = joblib.load(self.model_path)
            self.loaded_scaler = joblib.load(self.scaler_path)
        
        # Replicate features
        rooms_per_sqm = rooms / size
        is_studio = 1 if rooms == 1 else 0
        
        features = pd.DataFrame([[size, rooms, rooms_per_sqm, is_studio]], 
                              columns=['size', 'rooms', 'rooms_per_sqm', 'is_studio'])
        features_scaled = self.loaded_scaler.transform(features)
        return self.loaded_model.predict(features_scaled)[0]

if __name__ == "__main__":
    nm = NumericModel()
    nm.train()
