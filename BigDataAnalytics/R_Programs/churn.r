# Load necessary libraries
library(ggplot2)

# Load the churn modeling dataset from a CSV file
churn_data <- read.csv("C:/Users/jugal/Desktop/CODE/SEM7/BDA/Rprog/Churn_Modelling.csv")

# View the first few rows of the dataset
head(churn_data)

# Create a bar chart to visualize the distribution of churn (Exited) by Gender
bar_chart <- ggplot(churn_data, aes(x = Gender, fill = factor(Exited))) +
  geom_bar() +
  labs(title = "Churn Distribution by Gender", x = "Gender", y = "Count") +
  scale_fill_discrete(name = "Exited")

# Create a histogram to visualize the distribution of Age by churn status
histogram <- ggplot(churn_data, aes(x = Age, fill = factor(Exited))) +
  geom_histogram(binwidth = 5, alpha = 0.7) +
  labs(title = "Age Distribution by Churn Status", x = "Age", y = "Frequency") +
  scale_fill_discrete(name = "Exited")

# Create a scatterplot to visualize the relationship between Credit Score and Balance
scatterplot <- ggplot(churn_data, aes(x = CreditScore, y = Balance, color = factor(Exited))) +
  geom_point() +
  labs(title = "Scatterplot of Credit Score vs. Balance", x = "Credit Score", y = "Balance") +
  scale_color_discrete(name = "Exited")

# Create a box plot to visualize the distribution of Estimated Salary by churn status
box_plot <- ggplot(churn_data, aes(x = factor(Exited), y = EstimatedSalary, fill = factor(Exited))) +
  geom_boxplot() +
  labs(title = "Box Plot of Estimated Salary by Churn Status", x = "Churn Status", y = "Estimated Salary") +
  scale_fill_discrete(name = "Exited")

grid.arrange(scatterplot,bar_chart, histogram, box_plot, ncol = 2)
