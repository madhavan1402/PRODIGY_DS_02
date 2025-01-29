import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
gender_submission_df = pd.read_csv('gender_submission.csv')

# Display the first few rows of the training dataset
print("\nTraining Dataset Head:")
print(train_df.head())

# Basic dataset information
print("\nTraining Dataset Info:")
train_df.info()

print("\nMissing Values in Training Dataset:")
print(train_df.isnull().sum())

# Fill missing Age values with the median
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

# Fill missing Embarked values with the mode
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Drop the Cabin column due to too many missing values
train_df.drop(columns=['Cabin'], inplace=True)

# Verify missing values are handled
print("\nMissing Values After Cleaning:")
print(train_df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(train_df.describe())

# Exploratory Data Analysis
# Distribution of Survived
plt.figure()
sns.countplot(x='Survived', data=train_df, palette='pastel')
plt.title('Distribution of Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.savefig('distribution_of_survival.png')

# Survival by Gender
plt.figure()
sns.countplot(x='Survived', hue='Sex', data=train_df, palette='muted')
plt.title('Survival by Gender')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.savefig('survival_by_gender.png')

# Age distribution by Survival
plt.figure()
sns.histplot(data=train_df, x='Age', hue='Survived', kde=True, palette='cool')
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('age_distribution_by_survival.png')

# Survival by Pclass
plt.figure()
sns.countplot(x='Survived', hue='Pclass', data=train_df, palette='dark')
plt.title('Survival by Passenger Class')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.savefig('survival_by_pclass.png')

# Pairplot for numerical columns
sns.pairplot(train_df[['Survived', 'Age', 'Fare', 'Pclass']], hue='Survived', palette='husl')
plt.savefig('pairplot_survived.png')

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = train_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')

# Prepare the data for modeling (optional)
X = train_df.drop(columns=['Survived', 'Name', 'Ticket', 'PassengerId'])
y = train_df['Survived']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)

print("\nFeature Data Head:")
print(X.head())

# Save cleaned data for modeling
X.to_csv('cleaned_train_features.csv', index=False)
y.to_csv('cleaned_train_labels.csv', index=False)

print("\nData Cleaning and EDA Completed.")
