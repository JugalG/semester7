# Create a data frame for the existing salaries
employee_data <- data.frame(
  "Sr. No." = 1:10,
  "Name of employees" = c("Vivek", "Karan", "James", "Soham", "Renu", "Farah", "Hetal", "Mary", "Ganesh", "Krish"),
  Salaries = c(21000, 55000, 67000, 50000, 54000, 40000, 30000, 70000, 20000, 15000)
)

# Display the data frame
print(employee_data)

# Create a data frame for the new employees
new_employee_data <- data.frame(
  "Sr. No." = 11:15,
  "Name of employees" = c("Amit", "Sneha", "Rekha", "Rahul", "Priya"),
  Salaries = c(45000, 60000, 53000, 48000, 62000)
)

# Combine the new data with the existing data
combined_data <- rbind(employee_data, new_employee_data)

# Display the combined data
print(combined_data)

# Load necessary libraries for visualization
library(ggplot2)

# Create a histogram to visualize the distribution of salaries
ggplot(combined_data, aes(x = Salaries)) +
  geom_histogram(binwidth = 10000, fill = "blue", color = "black") +
  labs(title = "Salary Distribution", x = "Salaries", y = "Frequency") +
  theme_minimal()