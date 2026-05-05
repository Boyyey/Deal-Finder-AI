class FusionEngine:
    def __init__(self):
        pass

    def calculate_deal_score(self, actual_price, predicted_price, condition):
        # 1. Price Analysis
        price_ratio = actual_price / predicted_price
        price_per_sqm = actual_price / 1 # placeholder for size
        
        # Numerical Score (0 to 1)
        # We want a sigmoid-like curve or stepped scoring
        if price_ratio < 0.8: numeric_score = 1.0    # Massive bargain
        elif price_ratio < 0.95: numeric_score = 0.8 # Good deal
        elif price_ratio < 1.05: numeric_score = 0.5 # Fair market
        elif price_ratio < 1.2: numeric_score = 0.2  # Slightly overpriced
        else: numeric_score = 0.0                    # Bad deal

        # 2. Image Condition Score (Health of Property)
        condition_weights = {
            'luxury': 1.0,
            'normal': 0.6,
            'damaged': 0.0
        }
        image_score = condition_weights.get(condition, 0.5)

        # 3. Intelligent Fusion Logic (The "Laws")
        # A property is only a 'Great Deal' if it's reasonably priced AND in good health.
        # A cheap mansion in 'damaged' condition is 'Risky'.
        # A small home in 'luxury' condition at a fair price is 'Great'.
        
        # Weighted base score
        base_score = (0.6 * numeric_score) + (0.4 * image_score)
        
        # Override Laws
        if condition == 'damaged' and price_ratio < 0.8:
            deal_rating = "Risky Bargain"
            final_score = base_score * 0.8 # Penalty for condition
        elif condition == 'luxury' and price_ratio < 1.1:
            deal_rating = "Great Deal"
            final_score = min(1.0, base_score * 1.2) # Bonus for luxury
        elif price_ratio > 1.3:
            deal_rating = "Overpriced"
            final_score = base_score * 0.5
        elif condition == 'damaged':
            deal_rating = "Poor Value"
            final_score = base_score * 0.4
        else:
            if base_score > 0.7: deal_rating = "Good Deal"
            elif base_score > 0.4: deal_rating = "Fair Deal"
            else: deal_rating = "Bad Deal"
            final_score = base_score

        # 4. Detailed Explanations
        explanations = []
        
        # Price explanation
        if price_ratio < 0.9:
            explanations.append(f"Strong Value: Listing is {((1-price_ratio)*100):.1f}% below market average for this size.")
        elif price_ratio > 1.1:
            explanations.append(f"Premium Price: Listing is {((price_ratio-1)*100):.1f}% above market average.")
        else:
            explanations.append("Market Value: Price is aligned with comparable properties.")

        # Condition explanation
        if condition == 'luxury':
            explanations.append("High Health: Visuals indicate premium finishes and excellent maintenance.")
        elif condition == 'damaged':
            explanations.append("Low Health: Property shows signs of neglect or structural issues. High renovation costs expected.")
        else:
            explanations.append("Standard Health: Property appears well-maintained with standard features.")

        # Combined insight
        if condition == 'damaged' and price_ratio < 0.9:
            explanations.append("⚠️ Warning: The low price reflects the poor condition. Only suitable for investors/renovators.")
        elif condition == 'luxury' and price_ratio > 1.1:
            explanations.append("✨ Note: The premium price is likely due to the high-end luxury condition.")

        return {
            'rating': deal_rating,
            'score': float(final_score),
            'price_ratio': price_ratio,
            'condition': condition,
            'explanations': explanations
        }
