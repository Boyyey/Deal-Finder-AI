import pandas as pd
import numpy as np
import os
import requests
from PIL import Image, ImageDraw, ImageFont
import random

def download_real_images():
    """Download real property images for each category"""
    categories = {
        'luxury': [
            'https://picsum.photos/seed/luxury1/224/224.jpg',
            'https://picsum.photos/seed/luxury2/224/224.jpg',
            'https://picsum.photos/seed/luxury3/224/224.jpg',
            'https://picsum.photos/seed/luxury4/224/224.jpg',
            'https://picsum.photos/seed/luxury5/224/224.jpg'
        ],
        'normal': [
            'https://picsum.photos/seed/normal1/224/224.jpg',
            'https://picsum.photos/seed/normal2/224/224.jpg',
            'https://picsum.photos/seed/normal3/224/224.jpg',
            'https://picsum.photos/seed/normal4/224/224.jpg',
            'https://picsum.photos/seed/normal5/224/224.jpg'
        ],
        'damaged': [
            'https://picsum.photos/seed/damaged1/224/224.jpg',
            'https://picsum.photos/seed/damaged2/224/224.jpg',
            'https://picsum.photos/seed/damaged3/224/224.jpg',
            'https://picsum.photos/seed/damaged4/224/224.jpg',
            'https://picsum.photos/seed/damaged5/224/224.jpg'
        ]
    }
    
    os.makedirs('data/images', exist_ok=True)
    
    for category, urls in categories.items():
        for i, url in enumerate(urls):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img.save(f'data/images/{category}_{i+1}.jpg')
                    print(f"Downloaded {category}_{i+1}.jpg")
            except Exception as e:
                print(f"Failed to download {url}: {e}")

def generate_enhanced_property_data(num_samples=500):
    """Generate more realistic property data"""
    np.random.seed(42)
    
    # More realistic data distribution
    data = []
    
    for i in range(num_samples):
        # Size distribution with more realistic ranges
        if i < num_samples * 0.1:  # 10% luxury properties
            size = np.random.randint(180, 400)
            rooms = np.random.randint(4, 8)
            base_price_per_sqm = np.random.uniform(3500, 5000)
        elif i < num_samples * 0.3:  # 20% damaged properties
            size = np.random.randint(40, 150)
            rooms = np.random.randint(1, 3)
            base_price_per_sqm = np.random.uniform(800, 1500)
        else:  # 70% normal properties
            size = np.random.randint(60, 200)
            rooms = np.random.randint(1, 5)
            base_price_per_sqm = np.random.uniform(2000, 3000)
        
        # Add market noise and location factors
        location_factor = np.random.uniform(0.8, 1.3)
        market_noise = np.random.normal(0, 0.1)
        
        price = size * base_price_per_sqm * location_factor * (1 + market_noise)
        price = max(50000, min(price, 2000000))  # Clamp to realistic range
        
        # Assign condition based on price and size
        if base_price_per_sqm > 3500:
            condition = 'luxury'
        elif base_price_per_sqm < 1500:
            condition = 'damaged'
        else:
            condition = 'normal'
        
        data.append({
            'price': int(price),
            'size': size,
            'rooms': rooms,
            'condition': condition,
            'price_per_sqm': price / size,
            'year_built': np.random.randint(1950, 2023),
            'location_score': np.random.uniform(1, 10)
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/properties.csv', index=False)
    print(f"Generated {num_samples} enhanced property records")

if __name__ == "__main__":
    from io import BytesIO
    generate_enhanced_property_data()
    download_real_images()
