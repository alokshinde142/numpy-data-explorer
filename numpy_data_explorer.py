"""
NumPy Data Explorer
Author: Alok Shinde

Objective:
Explore and analyze structured sales data using NumPy.
Demonstrates:
- Array creation
- Indexing & slicing
- Mathematical operations
- Axis-wise & statistical analysis
- Reshaping & broadcasting
- Save & load operations
- Performance comparison
"""

import numpy as np
import time

np.random.seed(0)

# ----------------------------------
# 1. Create Structured Dataset
# Columns: Product_ID, Units_Sold, Price
# ----------------------------------

product_ids = np.arange(1, 11)
units_sold = np.random.randint(50, 200, size=10)
price = np.random.randint(100, 500, size=10)

data = np.column_stack((product_ids, units_sold, price))

print("\n📊 DATASET")
print("Columns: Product_ID | Units_Sold | Price")
print(data)

# ----------------------------------
# 2. Indexing & Slicing
# ----------------------------------

print("\n🔍 INDEXING & SLICING")
print("First 3 Products:\n", data[:3])
print("Units Sold Column:\n", data[:, 1])
print("Price Column:\n", data[:, 2])

# ----------------------------------
# 3. Revenue Calculation
# ----------------------------------

revenue = units_sold * price
print("\n💰 REVENUE PER PRODUCT")
print(revenue)

# ----------------------------------
# 4. Statistical Analysis
# ----------------------------------

print("\n📈 BUSINESS INSIGHTS")
print("Total Revenue:", np.sum(revenue))
print("Average Revenue:", np.mean(revenue))
print("Highest Revenue:", np.max(revenue))
print("Lowest Revenue:", np.min(revenue))

best_product = product_ids[np.argmax(revenue)]
print("Best Performing Product ID:", best_product)

# ----------------------------------
# 5. Reshaping Example
# ----------------------------------

reshaped = data.reshape(5, 6)
print("\n🔄 Reshaped Dataset (5x6):\n", reshaped)

# ----------------------------------
# 6. Save & Load
# ----------------------------------

np.save("sample_dataset.npy", data)
loaded_data = np.load("sample_dataset.npy")

print("\n💾 Data Saved & Reloaded Successfully")

# ----------------------------------
# 7. Performance Comparison
# ----------------------------------

print("\n⚡ PERFORMANCE COMPARISON")

size = 5_000_000

start = time.time()
py_list = [i * 2 for i in range(size)]
end = time.time()
print("Python List Time:", round(end - start, 4), "seconds")

start = time.time()
np_array = np.arange(size) * 2
end = time.time()
print("NumPy Array Time:", round(end - start, 4), "seconds")

print("\n✅ NumPy demonstrates faster vectorized computation.")
