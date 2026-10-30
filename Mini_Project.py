import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Generate sample data using CSV
np.random.seed(42)

with open("traffic_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Hour", "Vehicle_Count", "Average_Speed",
                     "Congestion"])

    for hour in range(24):
        for _ in range(10):
            vehicles = np.random.randint(20, 150)
            speed = np.random.randint(10, 80)

            if vehicles >= 100 or speed <= 20:
                congestion = "High"
            elif vehicles >= 60 or speed <= 35:
                congestion = "Medium"
            else:
                congestion = "Low"

            writer.writerow([hour, vehicles, speed, congestion])

# 2. Load and analyze data using Pandas
df = pd.read_csv("traffic_data.csv")

print("\nSMART TRAFFIC MONITORING SYSTEM")
print("--------------------------------")
print("Total records:", len(df))
print("Average vehicle count:", round(df["Vehicle_Count"].mean(), 2))
print("Average speed:", round(df["Average_Speed"].mean(), 2), "km/h")

# 3. SciPy: detect unusual vehicle counts
z_scores = np.abs(stats.zscore(df["Vehicle_Count"]))
print("Unusual traffic records:", int((z_scores > 2).sum()))

# 4. Logistic Regression: predict congestion category
X_class = df[["Hour", "Vehicle_Count", "Average_Speed"]]
y_class = df["Congestion"]

X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2,
    random_state=42, stratify=y_class
)

log_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
log_model.fit(X_train, y_train)

predictions = log_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Logistic Regression accuracy:",
      round(accuracy * 100, 2), "%")

# 5. Linear Regression: predict vehicle count
# Use hour and average speed to predict vehicle count
X_reg = df[["Hour", "Average_Speed"]]
y_reg = df["Vehicle_Count"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(Xr_train, yr_train)

volume_prediction = linear_model.predict(Xr_test)
mae = mean_absolute_error(yr_test, volume_prediction)

print("Linear Regression MAE:", round(mae, 2), "vehicles")

# 6. Graph 1: vehicle count by hour
plt.figure(figsize=(9, 5))
sns.lineplot(
    data=df, x="Hour", y="Vehicle_Count",
    estimator="mean", errorbar=None, marker="o"
)
plt.title("Average Traffic Volume by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Vehicle Count")
plt.xticks(range(24))
plt.tight_layout()
plt.savefig("traffic_by_hour.png")
plt.show()

# 7. Graph 2: speed versus vehicle count
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df, x="Vehicle_Count", y="Average_Speed",
    hue="Congestion"
)
plt.title("Vehicle Count vs Average Speed")
plt.tight_layout()
plt.savefig("traffic_speed_analysis.png")
plt.show()

# 8. Graph 3: congestion prediction
plt.figure(figsize=(9, 5))
sns.scatterplot(
    x=X_test["Vehicle_Count"],
    y=X_test["Average_Speed"],
    hue=predictions,
    palette="Set1"
)
plt.title("Logistic Regression: Predicted Congestion")
plt.xlabel("Vehicle Count")
plt.ylabel("Average Speed (km/h)")
plt.tight_layout()
plt.savefig("predicted_congestion.png")
plt.show()

# 9. Graph 4: actual versus predicted vehicle count
plt.figure(figsize=(8, 5))
plt.scatter(yr_test, volume_prediction)
plt.plot(
    [yr_test.min(), yr_test.max()],
    [yr_test.min(), yr_test.max()],
    linestyle="--"
)
plt.title("Linear Regression: Actual vs Predicted Traffic")
plt.xlabel("Actual Vehicle Count")
plt.ylabel("Predicted Vehicle Count")
plt.tight_layout()
plt.savefig("vehicle_count_prediction.png")
plt.show()

# 10. Graph 5: correlation between parameters
plt.figure(figsize=(7, 5))
sns.heatmap(
    df[["Hour", "Vehicle_Count", "Average_Speed"]].corr(),
    annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Traffic Parameter Correlation")
plt.tight_layout()
plt.savefig("traffic_correlation.png")
plt.show()

print("\nAnalysis completed. All traffic graphs generated.")

