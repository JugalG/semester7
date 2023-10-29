# Load necessary libraries
library(ggplot2)
library(dplyr)
# Load the loan dataset from a CSV file
loan_data <- read.csv("/Users/surajchavan/Downloads/Loan.csv")

# View the first few rows of the dataset
head(loan_data)


# Summary statistics for numeric columns
summary(loan_data[, c("ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term")])

# Summary statistics for categorical columns
summary(loan_data[, c("Gender", "Married", "Dependents", "Education", "Self_Employed", "Credit_History", "Property_Area", "Loan_Status")])

# Count of missing values
colSums(is.na(loan_data))

# Count of unique values for categorical columns
sapply(loan_data[, c("Gender", "Married", "Dependents", "Education", "Self_Employed", "Credit_History", "Property_Area", "Loan_Status")], function(x) length(unique(x)))

# Frequency table for Loan_Status
table(loan_data$Loan_Status)

# Correlation matrix for numeric variables
cor(loan_data[, c("ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term")])

# Crosstabulation of Loan_Status and other categorical variables
table(loan_data$Loan_Status, loan_data$Gender)
table(loan_data$Loan_Status, loan_data$Married)


# Create a scatterplot to visualize the relationship between ApplicantIncome and LoanAmount
scatterplot <- ggplot(loan_data, aes(x = ApplicantIncome, y = LoanAmount, color = Loan_Status)) +
  geom_point() +
  labs(title = "Scatterplot of Applicant Income vs. Loan Amount", x = "Applicant Income", y = "Loan Amount")

# Create a box plot to visualize the distribution of LoanAmount by Loan_Status
box_plot <- ggplot(loan_data, aes(x = Loan_Status, y = LoanAmount, fill = Loan_Status)) +
  geom_boxplot() +
  labs(title = "Box Plot of Loan Amount by Loan Status", x = "Loan Status", y = "Loan Amount")

# Create a bar chart to visualize the distribution of Loan_Status by Gender
bar_chart <- ggplot(loan_data, aes(x = Gender, fill = Loan_Status)) +
  geom_bar() +
  labs(title = "Loan Status Distribution by Gender", x = "Gender", y = "Count")

# Create a histogram to visualize the distribution of ApplicantIncome
histogram <- ggplot(loan_data, aes(x = ApplicantIncome, fill = Loan_Status)) +
  geom_histogram(binwidth = 1000, alpha = 0.7) +
  labs(title = "Histogram of Applicant Income", x = "Applicant Income", y = "Frequency")

grid.arrange(scatterplot,bar_chart, histogram, box_plot, ncol = 2)