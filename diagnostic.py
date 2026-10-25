import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "data" / "flipkart_com-ecommerce_sample.csv")

df["main_category"] = df["product_category_tree"].str.extract(r'\["([^>"]+?)\s*>>')[0]
print(df["main_category"].value_counts().head(15))

for mot in ["Smartphone", "Mobiles", "Laptops", "Headphones", "Refrigerator", "Watches"]:
    n = df["product_category_tree"].str.contains(mot, case=False, na=False).sum()
    print(f"{mot:<14}: {n} produits")