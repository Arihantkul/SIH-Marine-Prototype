import pandas as pd

try:
	# Load the CSV file
	df = pd.read_csv(r"C:\Users\aa\Desktop\Coding\csv\a.csv")
except FileNotFoundError:
	print("Error: The specified CSV file was not found.")
	exit(1)
except pd.errors.EmptyDataError:
	print("Error: The CSV file is empty.")
	exit(1)
except pd.errors.ParserError:
	print("Error: The CSV file is malformed.")
	exit(1)

# Show the first few rows
print("Full Data:\n", df)

# Show only first 3 rows
print("\nHead:\n", df.head(3))

# Show basic info
print("\nInfo:")
print(df.info())

# Show statistics (mean, min, max)
print("\nStatistics:\n", df.describe())

print(df.groupby("Species")["Count"].sum())
print(df.groupby("Region")["Count"].mean())



import matplotlib.pyplot as plt

df.groupby("Species")["Count"].sum().plot(kind="bar", color="skyblue")
plt.title("Fish Count by Species")
plt.ylabel("Total Count")
plt.show()
