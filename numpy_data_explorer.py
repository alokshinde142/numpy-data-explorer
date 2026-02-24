
"""
Project: NumPy Data Explorer
Author: Alok Shinde
Role Target: Data Analyst

Description:
This project demonstrates NumPy fundamentals applied to a structured sales dataset.
It includes:
- Array creation
- Indexing & slicing
- Mathematical operations
- Statistical & axis-wise analysis
- Reshaping
- Broadcasting
- Save & load operations
- Performance comparison
"""

import numpy as np
import time


def create_dataset():
    np.random.seed(42)

    product_ids = np.arange(1, 11)
    units_sold = np.random.randint(50, 200, size=10)
    price_per_unit = np.random.randint(100, 500, size=10)

    dataset = np.column_stack((product_ids, units_sold, price_per_unit))

    print("\nDATASET")
    print("Columns: Product_ID | Units_Sold | Price_per_Unit")
    print(dataset)

    return dataset, units_sold, price_per_unit, product_ids


def indexing_slicing(data):
    print("\nINDEXING & SLICING")
    print("First 3 rows:\n", data[:3])
    print("Units Sold column:\n", data[:, 1])
    print("Price column:\n", data[:, 2])


def revenue_analysis(units_sold, price_per_unit, product_ids):
    revenue = units_sold * price_per_unit

    print("\nREVENUE PER PRODUCT")
    print(revenue)

    print("\nBUSINESS INSIGHTS")
    print("Total Revenue:", np.sum(revenue))
    print("Average Revenue:", np.mean(revenue))
    print("Highest Revenue:", np.max(revenue))
    print("Lowest Revenue:", np.min(revenue))

    best_product = product_ids[np.argmax(revenue)]
    print("Best Performing Product ID:", best_product)

    return revenue


def reshape_example(data):
    print("\nRESHAPING")
    reshaped = data.reshape(5, 6)
    print("Reshaped Array (5x6):\n", reshaped)


def broadcasting_example(data):
    print("\nBROADCASTING")
    increase = np.array([0, 10, 20])
    print("After Increment:\n", data + increase)


def save_load_example(data):
    np.save("sample_dataset.npy", data)
    loaded = np.load("sample_dataset.npy")
    print("\nSAVE & LOAD SUCCESSFUL")
    return loaded


def performance_comparison():
    print("\nPERFORMANCE COMPARISON")

    size = 5_000_000

    start = time.time()
    py_list = [i * 2 for i in range(size)]
    end = time.time()
    print("Python List Time:", round(end - start, 4), "seconds")

    start = time.time()
    np_array = np.arange(size) * 2
    end = time.time()
    print("NumPy Array Time:", round(end - start, 4), "seconds")

    print("\nNumPy demonstrates faster vectorized computation.")


def main():
    data, units_sold, price_per_unit, product_ids = create_dataset()
    indexing_slicing(data)
    revenue_analysis(units_sold, price_per_unit, product_ids)
    reshape_example(data)
    broadcasting_example(data)
    save_load_example(data)
    performance_comparison()


if __name__ == "__main__":
    main()
