import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw

def generate_synthetic_data(num_samples=200):
    np.random.seed(42)
    
    # Numerical Data
    sizes = np.random.randint(40, 300, num_samples)  # sqm
    rooms = np.random.randint(1, 7, num_samples)
    
    # Base price calculation: $2000 per sqm + $10000 per room + noise
    base_prices = (sizes * 2500) + (rooms * 15000)
    noise = np.random.normal(0, 0.1, num_samples) * base_prices
    prices = base_prices + noise
    
    # Image labels (simulated)
    conditions = np.random.choice(['luxury', 'normal', 'damaged'], num_samples, p=[0.2, 0.6, 0.2])
    
    df = pd.DataFrame({
        'price': prices.astype(int),
        'size': sizes,
        'rooms': rooms,
        'condition': conditions
    })
    
    # Save CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/properties.csv', index=False)
    print(f"Generated {num_samples} records in data/properties.csv")

def create_dummy_images():
    os.makedirs('data/images', exist_ok=True)
    colors = {
        'luxury': (255, 215, 0),    # Gold
        'normal': (100, 149, 237),  # Cornflower Blue
        'damaged': (178, 34, 34)    # Firebrick
    }
    
    for label, color in colors.items():
        img = Image.new('RGB', (224, 224), color=color)
        draw = ImageDraw.Draw(img)
        draw.text((60, 100), label.upper(), fill=(255, 255, 255))
        img.save(f'data/images/template_{label}.png')
    print("Generated dummy image templates")

if __name__ == "__main__":
    generate_synthetic_data()
    create_dummy_images()
