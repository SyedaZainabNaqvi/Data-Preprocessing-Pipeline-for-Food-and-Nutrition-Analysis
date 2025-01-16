import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

# Initialize a global dataframe
df = pd.DataFrame()

# Function definitions for data preprocessing steps
def load_dataset():
    global df
    file_path = filedialog.askopenfilename()
    if file_path:
        df = pd.read_csv(file_path)
        clear_output()
        display_table(df)
        output_text.insert(tk.END, "Dataset loaded successfully!\n\n")


def display_table(data):
    # Clear previous table
    for row in tree.get_children():
        tree.delete(row)
    # Insert new table data
    tree["columns"] = list(data.columns)
    tree["show"] = "headings"
    for col in data.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)
    for index, row in data.iterrows():
        tree.insert("", "end", values=list(row))


def clear_output():
    output_text.delete(1.0, tk.END)


def handle_missing_values():
    global df
    clear_output()
    df = df.dropna()
    display_table(df)
    output_text.insert(tk.END, "Missing values handled.\n\n")


def remove_outliers():
    global df
    clear_output()
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    display_table(df)
    output_text.insert(tk.END, "Outliers removed.\n\n")


def normalize_data():
    global df
    clear_output()
    numeric_columns = df.select_dtypes(include=['number']).columns
    if not numeric_columns.empty:
        scaler = RobustScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        display_table(df)
        output_text.insert(tk.END, "Data normalized.\n\n")
    else:
        output_text.insert(tk.END, "No numeric columns to normalize.\n\n")


def encode_categorical_data():
    global df
    clear_output()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if not categorical_columns.empty:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        display_table(df)
        output_text.insert(tk.END, "Categorical data encoded.\n\n")
    else:
        output_text.insert(tk.END, "No categorical columns to encode.\n\n")


def plot_data_distribution():
    global df
    clear_output()
    output_text.insert(tk.END, "Displaying data distribution after scaling.\n\n")
    df.plot(kind='kde', subplots=True, layout=(3, 3), figsize=(12, 8), sharex=False)
    plt.tight_layout()
    plt.show()


def plot_boxplot_after_outliers():
    global df
    clear_output()
    output_text.insert(tk.END, "Displaying boxplot after removing outliers.\n\n")
    df.boxplot(figsize=(10, 6))
    plt.show()


def preprocess_all():
    global df
    clear_output()
    file_path = filedialog.askopenfilename(title="Select Cleaned Dataset for Preprocessing")
    if file_path:
        df = pd.read_csv(file_path)
        handle_missing_values()
        remove_outliers()
        normalize_data()
        encode_categorical_data()
        display_table(df)
        output_text.insert(tk.END, "All preprocessing steps completed and cleaned data displayed.\n\n")

# GUI Implementation
root = tk.Tk()
root.title("Data Preprocessing GUI")
root.geometry("1200x800")  # Increased window size for better layout

# Output Text Area
output_text = tk.Text(root, wrap=tk.WORD, height=10, width=150, font=("Arial", 10), bg="lightgray")
output_text.pack(pady=10)

# Table for Displaying Data
tree_frame = tk.Frame(root)
tree_frame.pack(pady=10)
tree = ttk.Treeview(tree_frame)
tree.pack()

# Button Frame
button_frame = tk.Frame(root, bg="black")
button_frame.pack(pady=10)

# Arrange buttons in rows with three per row
buttons = [
    {"text": "Load Dataset", "command": load_dataset, "bg": "#003366"},
    {"text": "Handle Missing Values", "command": handle_missing_values, "bg": "#336600"},
    {"text": "Remove Outliers", "command": remove_outliers, "bg": "#FF6600"},
    {"text": "Normalize Data", "command": normalize_data, "bg": "#660066"},
    {"text": "Encode Categorical Data", "command": encode_categorical_data, "bg": "#006666"},
    {"text": "Data Distribution After Scaling", "command": plot_data_distribution, "bg": "#990033"},
    {"text": "Boxplot After Removing Outliers", "command": plot_boxplot_after_outliers, "bg": "#660000"},
    {"text": "Preprocess All", "command": preprocess_all, "bg": "#006600"},
]

for i in range(0, len(buttons), 3):
    row_frame = tk.Frame(button_frame, bg="black")
    row_frame.pack(pady=5)
    for button in buttons[i:i+3]:
        tk.Button(
            row_frame, 
            text=button["text"], 
            command=button["command"], 
            bg=button["bg"], 
            fg="white", 
            font=("Arial", 10, "bold"),
            width=25, 
            height=2
        ).pack(side=tk.LEFT, padx=10)

root.mainloop()