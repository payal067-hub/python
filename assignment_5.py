import pandas as pd

df = pd.read_csv("E:/CODE/Python/final_student_data.csv")

print("=" * 100)
print(df)
print("=" * 100)
print("1.FILTERING")

high_paid_employee = df[(df["Department"] == "IT") & (df["Salary"]>25000)]
print("## For IT Department \n",high_paid_employee)
print("=" * 100)

high_paid_employee = df[(df["Department"] == "Computer") & (df["Salary"]>25000)]
print("## For Computer Department \n",high_paid_employee)
print("=" * 100)

high_paid_employee = df[(df["Department"] == "Electronics") & (df["Salary"]>25000)]
print("## For Electronics Department \n",high_paid_employee)
print("=" * 100)

df = df.sort_values(["Age","Salary"], ascending=[True,False])
print("2.SORTING")
print(df)
print("=" * 100)


df = df.groupby("Department")["Salary"].mean()
print("3.CALCULATING GROUPBY DATA PRESENT IN COLUMN")
print(df)
print("=" * 100)

