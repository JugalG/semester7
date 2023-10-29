# Vector of values
values <- c(23, 45, 10, 34, 89, 20, 67, 99)

# Sort the vector in ascending order
sorted_values_asc <- sort(values)

# Sort the vector in descending order
sorted_values_desc <- sort(values, decreasing = TRUE)

# Create a data frame for visualization
data_df <- data.frame(
  Values = c(sorted_values_asc, sorted_values_desc),
  Order = rep(c("Ascending", "Descending"), each = length(values))
)

# Load necessary libraries
library(ggplot2)

# Create a line plot for visualization
line_plot <- ggplot(data_df, aes(x = 1:length(Values), y = Values, group = Order, color = Order)) +
  geom_line() +
  geom_point() +
  labs(title = "Sorted Values in Ascending and Descending Order", x = "Index", y = "Value") +
  scale_color_manual(values = c("Ascending" = "blue", "Descending" = "red"))

# Display the line plot
print(line_plot)