# Load necessary libraries
library(ggplot2)
library(dplyr)

# Load the supermarket sales dataset from a CSV file
supermarket_data <- read.csv("/Users/surajchavan/Downloads/Supermarket2.csv")

# View the first few rows of the dataset
head(supermarket_data)

# Summary statistics for numeric columns
summary(supermarket_data[, c("Unit_price", "Quantity", "Tax", "Total", "cogs", "gross_margin_percentage", "gross_income", "Rating")])

# Summary statistics for categorical columns
summary(supermarket_data[, c("Branch", "City", "Customer_type", "Gender", "Product_line", "Payment")])

# Count of missing values
colSums(is.na(supermarket_data))

# Count of unique values for categorical columns
sapply(supermarket_data[, c("Branch", "City", "Customer_type", "Gender", "Product_line", "Payment")], function(x) length(unique(x)))

# Frequency table for Product_line
table(supermarket_data$Product_line)


# Create a scatterplot to visualize the relationship between Unit_price and Quantity
scatterplot <- ggplot(supermarket_data, aes(x = Unit_price, y = Quantity, color = Customer_type)) +
  geom_point() +
  labs(title = "Scatterplot of Unit Price vs. Quantity", x = "Unit Price", y = "Quantity")

# Create a box plot to visualize the distribution of Total by City
box_plot <- ggplot(supermarket_data, aes(x = City, y = Total, fill = City)) +
  geom_boxplot() +
  labs(title = "Box Plot of Total Sales by City", x = "City", y = "Total Sales")

# Create a bar chart to visualize the distribution of Customer_type
bar_chart <- ggplot(supermarket_data, aes(x = Customer_type, fill = Customer_type)) +
  geom_bar() +
  labs(title = "Customer Type Distribution", x = "Customer Type", y = "Count")

# Additional analyses and visualizations can be performed based on specific research questions and objectives.
# Additional visualizations and analyses can be performed based on your specific research questions and objectives.
grid.arrange(scatterplot,bar_chart,  box_plot, ncol = 2)