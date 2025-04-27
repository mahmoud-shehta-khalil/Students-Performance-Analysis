import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import os

# Get the directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Define sigmoid function for MachineLearning class
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# DataProcessor Class
class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def loadData(self):
        try:
            self.data = pd.read_csv(self.file_path)
            # Strip whitespace from column names to avoid mismatches
            self.data.columns = self.data.columns.str.strip()
            return self.data
        except FileNotFoundError:
            messagebox.showerror("Error", "File not found!")
            return None

    def dropColumns(self):
        # Drop irrelevant columns by name
        columns_to_drop = [
            'Timestamp', 'Email Address', 'Full Name',
            'If yes, mention them below:', 'Any Suggestions for us:', 'Email'
        ]
        # Only drop columns that exist in the DataFrame
        columns_to_drop = [col for col in columns_to_drop if col in self.data.columns]
        self.data.drop(columns=columns_to_drop, inplace=True)
        return self.data


# BasicDataProcessor Class
class BasicDataProcessor(DataProcessor):
    def __init__(self, file_path):
        super().__init__(file_path)

    def handleMissingValues(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return None
        for column in self.data.columns:
            if self.data[column].isna().sum() > 0:  # Only process columns with missing values
                if self.data[column].dtype == 'object':
                    # Categorical: Fill with most frequent value
                    most_frequent = self.data[column].mode()[0]
                    self.data[column] = self.data[column].fillna(most_frequent)
                else:
                    # Numerical: Fill with mean
                    mean_value = self.data[column].mean()
                    self.data[column] = self.data[column].fillna(mean_value)
        return self.data

    def labelEncoder(self):
        label_encoder = LabelEncoder()
        required_columns = [
            ('Gender', 'Gender_numeric'),
            ('Current Year of Study:', 'Current_Year_of_Study_Numeric'),
            ('Branch of Study:', 'Branch_of_Study_Numeric'),
            ('Average Hours of Daily Study;', 'Average_Hours_Numeric'),
            ('Participation in Extra-curricular Activities:', 'Extra-curricular_Numeric'),
            ('Studying Ambiance Preferences', 'Study_Ambiance_Preferences_Numeric')
        ]
        for col, new_col in required_columns:
            if col in self.data.columns:
                self.data[new_col] = label_encoder.fit_transform(self.data[col])
            else:
                messagebox.showerror("Error", f"Column '{col}' not found!")
        return self.data


# DataWrangling Class
class DataWrangling(BasicDataProcessor):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.loadData()
        self.dropColumns()
        self.handleMissingValues()
        self.labelEncoder()

    def pieChart(self, frame):
        if 'Branch of Study:' in self.data.columns:
            branch_counts = self.data['Branch of Study:'].value_counts()
            fig = Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            ax.pie(branch_counts, labels=branch_counts.index, autopct='%1.1f%%', startangle=140)
            ax.set_title('Distribution of Branch of Study')
            ax.axis('equal')
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            return canvas
        else:
            messagebox.showerror("Error", "Column 'Branch of Study:' not found!")
            return None

    def branchSubsets(self):
        if 'Branch of Study:' in self.data.columns:
            self.unique_branches = self.data['Branch of Study:'].unique()
            self.branch_subsets = {}
            for branch in self.unique_branches:
                self.branch_subsets[branch] = self.data[self.data['Branch of Study:'] == branch]
        else:
            messagebox.showerror("Error", "Column 'Branch of Study:' not found!")


# Segmentation Class
class Segmentation(DataWrangling):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.unique_years = None
        self.years_subsets = None
        self.average_hours_of_study_by_branch = None

    def pieChartForBranches(self, frame):
        if not self.branch_subsets:
            messagebox.showerror("Error", "No branch subsets available!")
            return None
        fig = Figure(figsize=(12, 8))
        num_branches = len(self.branch_subsets)
        num_cols = 2
        num_rows = (num_branches + 1) // 2
        legend_labels = ['First Year', 'Second Year', 'Third Year', 'Final Year']
        for i, (branch, subset) in enumerate(self.branch_subsets.items(), 1):
            if 'Current Year of Study:' in subset.columns:
                year_counts = subset['Current Year of Study:'].value_counts()
                ax = fig.add_subplot(num_rows, num_cols, i)
                wedges, texts, autotexts = ax.pie(year_counts, labels=None, autopct=lambda pct: f'{pct:.0f}%',
                                                  startangle=140)
                ax.set_title(f'{branch}', fontsize=10)
                ax.axis('equal')
            else:
                messagebox.showerror("Error", "Column 'Current Year of Study:' not found!")
                return None
        fig.legend(wedges, legend_labels, loc='center right')
        fig.subplots_adjust(right=0.85)
        for autotext in autotexts:
            autotext.set_fontsize(8)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        return canvas

    def uniqueYears(self):
        if 'Current Year of Study:' in self.data.columns:
            self.unique_years = self.data['Current Year of Study:'].unique()
            self.years_subsets = {}
            for year in self.unique_years:
                self.years_subsets[year] = self.data[self.data['Current Year of Study:'] == year]
        else:
            messagebox.showerror("Error", "Column 'Current Year of Study:' not found!")

    def pieChartForYears(self, frame):
        if not self.years_subsets:
            messagebox.showerror("Error", "No year subsets available!")
            return None
        fig = Figure(figsize=(12, 8))
        num_years = len(self.years_subsets)
        num_cols = 2
        num_rows = (num_years + 1) // 2
        for i, (year, subset) in enumerate(self.years_subsets.items(), 1):
            if 'Branch of Study:' in subset.columns:
                branch_counts = subset['Branch of Study:'].value_counts()
                ax = fig.add_subplot(num_rows, num_cols, i)
                wedges, texts, autotexts = ax.pie(branch_counts, labels=None, autopct='%1.1f%%', startangle=140)
                ax.set_title(f'Distribution of Current Year of Study for {year}', fontsize=10)
                ax.axis('equal')
            else:
                messagebox.showerror("Error", "Column 'Branch of Study:' not found!")
                return None
        fig.subplots_adjust(right=0.85)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        return canvas

    def averageStudyHoursByBranch(self, frame):
        if not self.branch_subsets:
            messagebox.showerror("Error", "No branch subsets available!")
            return None
        self.average_hours_of_study_by_branch = {}
        for branch, data in self.branch_subsets.items():
            if 'Average_Hours_Numeric' in data.columns:
                self.average_hours_of_study_by_branch[branch] = data['Average_Hours_Numeric'].mean()
            else:
                messagebox.showerror("Error", "Column 'Average_Hours_Numeric' not found!")
                return None
        branches = list(self.average_hours_of_study_by_branch.keys())
        average_hours = list(self.average_hours_of_study_by_branch.values())
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.bar(range(len(branches)), average_hours, color='orange')  # Use range for x positions
        ax.set_xlabel('Branch of Study')
        ax.set_ylabel('Average Hours of Study')
        ax.set_title('Average Hours of Study by Branch')
        # Explicitly set the ticks and their labels
        ax.set_xticks(range(len(branches)))  # Set tick positions
        ax.set_xticklabels(branches, rotation=45, ha='right')  # Set tick labels
        for i in range(len(branches)):
            ax.text(i, average_hours[i] + 0.05, f'{average_hours[i]:.2f}', ha='center', va='bottom')
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        return canvas

    def averageCGPAByBranch(self, frame):
        if not self.branch_subsets:
            messagebox.showerror("Error", "No branch subsets available!")
            return None
        average_cgpa_by_branch = {}
        for branch, data in self.branch_subsets.items():
            if 'Current CGPA (0.0 - 10.0):' in data.columns:
                average_cgpa_by_branch[branch] = data['Current CGPA (0.0 - 10.0):'].mean()
            else:
                messagebox.showerror("Error", "Column 'Current CGPA (0.0 - 10.0):' not found!")
                return None
        branches = list(average_cgpa_by_branch.keys())
        average_cgpa = list(average_cgpa_by_branch.values())
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.bar(range(len(branches)), average_cgpa, color='green')  # Use range for x positions
        ax.set_xlabel('Branch of Study')
        ax.set_ylabel('Average CGPA')
        ax.set_title('Average CGPA by Branch')
        # Explicitly set the ticks and their labels
        ax.set_xticks(range(len(branches)))  # Set tick positions
        ax.set_xticklabels(branches, rotation=45, ha='right')  # Set tick labels
        for i in range(len(branches)):
            ax.text(i, average_cgpa[i] + 0.05, f'{average_cgpa[i]:.2f}', ha='center', va='bottom')
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        return canvas


# MachineLearning Class
class MachineLearning(Segmentation):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.W = None
        self.B = None
        self.cost_list = None
        self.generate_train_test_files()
        # Debug: Check class distribution
        print("Class distribution of Overall Performance:")
        print(self.data['Overall Performance'].value_counts())

    def generate_train_test_files(self):
        """
        Generate training and test files, including feature scaling.
        """
        if self.data is None or 'Overall Performance' not in self.data.columns:
            messagebox.showerror("Error", "Column 'Overall Performance' not found or no data loaded!")
            raise KeyError("Column 'Overall Performance' not found")
        self.feature_columns = [
            'Gender_numeric', 'Current_Year_of_Study_Numeric', 'Branch_of_Study_Numeric',
            'Average_Hours_Numeric', 'Extra-curricular_Numeric', 'Study_Ambiance_Preferences_Numeric'
        ]
        missing_features = [col for col in self.feature_columns if col not in self.data.columns]
        if missing_features:
            messagebox.showerror("Error", f"Missing feature columns: {missing_features}")
            raise KeyError(f"Missing feature columns: {missing_features}")
        X = self.data[self.feature_columns]
        y = self.data['Overall Performance'].astype(int)

        # Apply feature scaling (standardization)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

        X = X.copy()
        X['Id'] = range(len(X))
        y_df = pd.DataFrame({'Id': range(len(y)), 'Target': y})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train.to_csv(os.path.join(BASE_DIR, 'train_X.csv'), index=False)
        pd.DataFrame({'Id': X_train['Id'], 'Target': y_train}).to_csv(os.path.join(BASE_DIR, 'train_Y.csv'),
                                                                      index=False)
        X_test.to_csv(os.path.join(BASE_DIR, 'test_X.csv'), index=False)
        pd.DataFrame({'Id': X_test['Id'], 'Target': y_test}).to_csv(os.path.join(BASE_DIR, 'test_Y.csv'), index=False)

        self.X_train = pd.read_csv(os.path.join(BASE_DIR, 'train_X.csv'))
        self.Y_train = pd.read_csv(os.path.join(BASE_DIR, 'train_Y.csv'))
        self.X_test = pd.read_csv(os.path.join(BASE_DIR, 'test_X.csv'))
        self.Y_test = pd.read_csv(os.path.join(BASE_DIR, 'test_Y.csv'))

    def dropID(self):
        """
        Remove the 'Id' column from the training and test sets and convert them to NumPy arrays.
        If the data is already a NumPy array, skip the drop operation.
        Returns:
            Tuple of (X_train, Y_train, X_test, Y_test) as NumPy arrays.
        """
        # Check if X_train is a DataFrame; if so, drop the 'Id' column
        if isinstance(self.X_train, pd.DataFrame):
            self.X_train = self.X_train.drop("Id", axis=1)
            self.Y_train = self.Y_train.drop("Id", axis=1)
            self.X_test = self.X_test.drop("Id", axis=1)
            self.Y_test = self.Y_test.drop("Id", axis=1)
            # Convert to NumPy arrays
            self.X_train = self.X_train.values
            self.Y_train = self.Y_train.values
            self.X_test = self.X_test.values
            self.Y_test = self.Y_test.values
            # Reshape for model training
            self.X_train = self.X_train.T
            self.Y_train = self.Y_train.reshape(1, self.X_train.shape[1])
            self.X_test = self.X_test.T
            self.Y_test = self.Y_test.reshape(1, self.X_test.shape[1])
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    def model(self, learning_rate, iterations):
        """
        Train the logistic regression model using gradient descent with class weighting.
        Args:
            learning_rate: Learning rate for gradient descent.
            iterations: Number of iterations for training.
        Returns:
            Tuple of (weights, bias, cost_list).
        """
        m = self.X_train.shape[1]
        n = self.X_train.shape[0]
        self.W = np.zeros((n, 1))
        self.B = 0
        self.cost_list = []

        # Compute class weights based on class distribution
        class_0_count = np.sum(self.Y_train == 0)
        class_1_count = np.sum(self.Y_train == 1)
        weight_0 = m / (2 * class_0_count)  # Weight for class 0
        weight_1 = m / (2 * class_1_count)  # Weight for class 1

        for i in range(iterations):
            Z = np.dot(self.W.T, self.X_train) + self.B
            A = sigmoid(Z)
            # Apply class weights to the cost function
            cost = -(1 / m) * np.sum(
                weight_1 * self.Y_train * np.log(A + 1e-15) +
                weight_0 * (1 - self.Y_train) * np.log(1 - A + 1e-15)
            )
            # Apply class weights to gradients
            error = A - self.Y_train
            weighted_error = np.where(self.Y_train == 1, weight_1 * error, weight_0 * error)
            dW = (1 / m) * np.dot(weighted_error, self.X_train.T)
            dB = (1 / m) * np.sum(weighted_error)
            self.W = self.W - learning_rate * dW.T
            self.B = self.B - learning_rate * dB
            self.cost_list.append(cost)
            if i % (iterations // 10) == 0:
                print(f"Cost after {i} iteration is: {cost}")
        return self.W, self.B, self.cost_list

    def curve(self, frame):
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(np.arange(len(self.cost_list)), self.cost_list)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Cost')
        ax.set_title('Cost vs Iterations')
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        return canvas

    # Inside MachineLearning class

    def plot_comparison(self, frame, train_test_acc, k_fold_acc):
        """
        Plot a bar chart comparing Train-Test Split accuracy and K-Fold CV accuracy.
        Args:
            frame: The tkinter frame to display the plot in.
            train_test_acc: Accuracy from Train-Test Split.
            k_fold_acc: Average accuracy from K-Fold Cross-Validation.
        Returns:
            The canvas containing the plot.
        """
        # Create the bar chart
        accuracies = [train_test_acc, k_fold_acc]
        labels = ['Train-Test Split', 'K-Fold CV']
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        bars = ax.bar(labels, accuracies, color=['lightgreen', 'lightcoral'])
        ax.set_title('Accuracy Comparison')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy (%)')

        # Add text labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval * 100:.2f}%', ha='center', va='bottom')

        # Display the plot in the frame
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        return canvas

    def accuracy(self, frame):
        if self.X_test is None or self.Y_test is None or self.W is None or self.B is None:
            messagebox.showerror("Error", "Model not trained or data not loaded!")
            return None
        Z = np.dot(self.W.T, self.X_test) + self.B
        A = sigmoid(Z)
        A = A > 0.3  # Lower the threshold to 0.3
        A = np.array(A, dtype='int64')
        Y_test_flat = self.Y_test.flatten()
        A_flat = A.flatten()

        acc = accuracy_score(Y_test_flat, A_flat)
        precision = precision_score(Y_test_flat, A_flat, zero_division=0)
        recall = recall_score(Y_test_flat, A_flat)
        f1 = f1_score(Y_test_flat, A_flat)

        label_acc = ttkb.Label(frame, text=f"Accuracy: {acc * 100:.2f}%", font=("Helvetica", 12))
        label_acc.pack(pady=5)
        label_precision = ttkb.Label(frame, text=f"Precision: {precision:.2f}", font=("Helvetica", 12))
        label_precision.pack(pady=5)
        label_recall = ttkb.Label(frame, text=f"Recall: {recall:.2f}", font=("Helvetica", 12))
        label_recall.pack(pady=5)
        label_f1 = ttkb.Label(frame, text=f"F1-Score: {f1:.2f}", font=("Helvetica", 12))
        label_f1.pack(pady=5)

        return label_acc, label_precision, label_recall, label_f1

    def plot_metrics(self, frame):
        if self.X_test is None or self.Y_test is None or self.W is None or self.B is None:
            messagebox.showerror("Error", "Model not trained or data not loaded!")
            return None

        Z = np.dot(self.W.T, self.X_test) + self.B
        A = sigmoid(Z)
        A = A > 0.3  # Lower the threshold to 0.3
        A = np.array(A, dtype='int64')
        Y_test_flat = self.Y_test.flatten()
        A_flat = A.flatten()

        acc = accuracy_score(Y_test_flat, A_flat)
        precision = precision_score(Y_test_flat, A_flat, zero_division=0)
        recall = recall_score(Y_test_flat, A_flat)
        f1 = f1_score(Y_test_flat, A_flat)

        metrics = [acc, precision, recall, f1]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        bars = ax.bar(metric_names, metrics, color='skyblue')
        ax.set_title('Model Evaluation Metrics')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        return canvas

    def k_fold_cross_validation(self, k=5, learning_rate=0.0015, iterations=100000):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded!")
            return None
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        accuracies = []
        for train_index, test_index in kf.split(self.data):
            X_train = self.data.iloc[train_index][self.feature_columns].values.T
            Y_train = self.data.iloc[train_index]['Overall Performance'].values.reshape(1, -1)
            X_test = self.data.iloc[test_index][self.feature_columns].values.T
            Y_test = self.data.iloc[test_index]['Overall Performance'].values.reshape(1, -1)

            self.W = np.zeros((X_train.shape[0], 1))
            self.B = 0
            prev_cost = float('inf')  
            for i in range(iterations):
                Z = np.dot(self.W.T, X_train) + self.B
                A = sigmoid(Z)
                cost = -(1 / X_train.shape[1]) * np.sum(
                    Y_train * np.log(A + 1e-15) + (1 - Y_train) * np.log(1 - A + 1e-15))

                # Early stopping check
                if abs(prev_cost - cost) < 0.0001:
                    print(f"Early stopping at iteration {i}, cost change: {abs(prev_cost - cost):.6f}")
                    break
                prev_cost = cost

                dW = (1 / X_train.shape[1]) * np.dot(A - Y_train, X_train.T)
                dB = (1 / X_train.shape[1]) * np.sum(A - Y_train)
                self.W = self.W - learning_rate * dW.T
                self.B = self.B - learning_rate * dB

            Z_test = np.dot(self.W.T, X_test) + self.B
            A_test = sigmoid(Z_test)
            A_test = A_test > 0.3  # Lower the threshold to 0.3
            A_test = np.array(A_test, dtype='int64')
            acc = accuracy_score(Y_test.flatten(), A_test.flatten())
            precision = precision_score(Y_test.flatten(), A_test.flatten(), zero_division=0)
            accuracies.append(acc)

        avg_accuracy = np.mean(accuracies)
        return avg_accuracy

# Main Application Class
class StudentPerformanceApp(ttkb.Window):
    def __init__(self, file_path):
        super().__init__(title="Student Performance Analysis", themename="flatly")
        self.geometry("1300x800")
        self.file_path = file_path
        self.current_page = None
        self.canvas = None
        self.model = MachineLearning(file_path)

        # Navigation Bar
        self.nav_frame = ttkb.Frame(self)
        self.nav_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        pages = [
            ("Branch Distribution", self.show_page1),
            ("Year by Branch", self.show_page2),
            ("Branch by Year", self.show_page3),
            ("Study Hours", self.show_page4),
            ("CGPA by Branch", self.show_page5),
            ("ML Model", self.show_page6),
            ("Comparison", self.show_page7),  # New page for comparison
            ("Evaluation", self.show_page8)  # New page for evaluation
        ]
        for text, command in pages:
            btn = ttkb.Button(self.nav_frame, text=text, command=command, bootstyle=INFO)
            btn.pack(side=tk.LEFT, padx=5)

        # Content Frame
        self.content_frame = ttkb.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        self.label = ttkb.Label(self.content_frame, text="Welcome to Student Performance Analysis",
                                font=("Helvetica", 20), bootstyle=PRIMARY)
        self.label.pack(pady=20)

        # Display first page by default
        self.show_page1()

    def clear_content(self):
        for widget in self.content_frame.winfo_children():
            if widget != self.label:
                widget.destroy()
        self.canvas = None

    def show_page1(self):
        self.clear_content()
        self.current_page = 1
        self.canvas = self.model.pieChart(self.content_frame)

    def show_page2(self):
        self.clear_content()
        self.current_page = 2
        self.model.branchSubsets()
        self.canvas = self.model.pieChartForBranches(self.content_frame)

    def show_page3(self):
        self.clear_content()
        self.current_page = 3
        self.model.uniqueYears()
        self.canvas = self.model.pieChartForYears(self.content_frame)

    def show_page4(self):
        self.clear_content()
        self.current_page = 4
        self.model.branchSubsets()
        self.canvas = self.model.averageStudyHoursByBranch(self.content_frame)

    def show_page5(self):
        self.clear_content()
        self.current_page = 5
        self.model.branchSubsets()
        self.canvas = self.model.averageCGPAByBranch(self.content_frame)

    def show_page6(self):
        self.clear_content()
        self.current_page = 6
        iterations = 100000
        learning_rate = 0.01  # Increased learning rate
        if self.model.W is None or self.model.B is None:
            self.model.dropID()
            self.model.model(learning_rate, iterations)
        self.canvas = self.model.curve(self.content_frame)
        labels = self.model.accuracy(self.content_frame)

        k_fold_accuracy = self.model.k_fold_cross_validation(k=5, learning_rate=learning_rate, iterations=iterations)
        if k_fold_accuracy is not None:
            Z = np.dot(self.model.W.T, self.model.X_test) + self.model.B
            A = sigmoid(Z)
            A = A > 0.3
            A = np.array(A, dtype='int64')
            acc = accuracy_score(self.model.Y_test.flatten(), A.flatten())

            comparison_text = f"Train-Test Split Accuracy: {acc * 100:.2f}%\n" \
                              f"K-Fold CV Average Accuracy: {k_fold_accuracy * 100:.2f}%"
            label_comparison = ttkb.Label(self.content_frame, text=comparison_text, font=("Helvetica", 12))
            label_comparison.pack(pady=10)

    def show_page7(self):
        self.clear_content()
        self.current_page = 7
        iterations = 100000
        learning_rate = 0.01  # Increased learning rate
        if self.model.W is None or self.model.B is None:
            self.model.dropID()
            self.model.model(learning_rate, iterations)

        Z = np.dot(self.model.W.T, self.model.X_test) + self.model.B
        A = sigmoid(Z)
        A = A > 0.3
        A = np.array(A, dtype='int64')
        train_test_acc = accuracy_score(self.model.Y_test.flatten(), A.flatten())

        k_fold_acc = self.model.k_fold_cross_validation(k=5, learning_rate=learning_rate, iterations=iterations)

        if k_fold_acc is not None:
            chart_frame = ttkb.Frame(self.content_frame)
            chart_frame.pack(fill=tk.BOTH, expand=True)
            self.canvas = self.model.plot_comparison(chart_frame, train_test_acc, k_fold_acc)
            text_frame = ttkb.Frame(self.content_frame)
            text_frame.pack(fill=tk.X, pady=10)
            comparison_text = f"Train-Test Split Accuracy: {train_test_acc * 100:.2f}%\n" \
                              f"K-Fold CV Average Accuracy: {k_fold_acc * 100:.2f}%"
            label_comparison = ttkb.Label(text_frame, text=comparison_text, font=("Helvetica", 12))
            label_comparison.pack(pady=10)

    def show_page8(self):
        self.clear_content()
        self.current_page = 8
        iterations = 100000
        learning_rate = 0.01  # Increased learning rate
        if self.model.W is None or self.model.B is None:
            self.model.dropID()
            self.model.model(learning_rate, iterations)

        chart_frame = ttkb.Frame(self.content_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = self.model.plot_metrics(chart_frame)
        text_frame = ttkb.Frame(self.content_frame)
        text_frame.pack(fill=tk.X, pady=10)
        labels = self.model.accuracy(text_frame)

if __name__ == "__main__":
    file_path = os.path.join(BASE_DIR, "Student Responses.csv")
    app = StudentPerformanceApp(file_path)
    app.mainloop()