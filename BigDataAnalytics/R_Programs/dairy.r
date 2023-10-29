# Load necessary libraries
library(ggplot2)

# Create a data frame for the sales data
sales_data <- data.frame(
  Week = 1:5,
  Bread = c(12, 3, 5, 11, 9),
  Milk = c(21, 27, 18, 20, 15),
  Cola = c(10, 1, 33, 6, 12),
  Chocolate = c(6, 7, 4, 13, 12),
  Detergent = c(5, 8, 12, 20, 23)
)

# Create a line plot to visualize sales over time
ggplot(sales_data, aes(x = Week)) +
  geom_line(aes(y = Bread, color = "Bread"), linewidth = 1) +
  geom_line(aes(y = Milk, color = "Milk"),linewidth = 1) +
  geom_line(aes(y = Cola, color = "Cola"), linewidth = 1) +
  geom_line(aes(y = Chocolate, color = "Chocolate"), linewidth = 1) +
  geom_line(aes(y = Detergent, color = "Detergent"), linewidth = 1) +
  labs(title = "Product Sales Over Time", x = "Week", y = "Sales") +
  scale_color_manual(values = c("Bread" = "blue", "Milk" = "green", "Cola" = "red", "Chocolate" = "purple", "Detergent" = "orange")) +
  theme_minimal()