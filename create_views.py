import sqlite3
import os

# Path to database
db_path = os.path.join(os.path.dirname(__file__), "data", "northwind.sqlite")

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Creating compatibility views...")

# Create lowercase views for simpler SQL generation
views = [
    "CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM Orders;",
    "CREATE VIEW IF NOT EXISTS order_items AS SELECT * FROM \"Order Details\";",
    "CREATE VIEW IF NOT EXISTS products AS SELECT * FROM Products;",
    "CREATE VIEW IF NOT EXISTS customers AS SELECT * FROM Customers;",
    "CREATE VIEW IF NOT EXISTS categories AS SELECT * FROM Categories;",
    "CREATE VIEW IF NOT EXISTS suppliers AS SELECT * FROM Suppliers;"
]

for view_sql in views:
    try:
        cursor.execute(view_sql)
        print(f"✓ Created: {view_sql.split()[5]}")
    except Exception as e:
        print(f"✗ Failed: {e}")

conn.commit()
conn.close()

print("\n✅ Compatibility views created successfully!")
print("You can now use: orders, order_items, products, customers, categories, suppliers")
