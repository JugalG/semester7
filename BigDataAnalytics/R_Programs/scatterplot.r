# Load necessary libraries
library(ggplot2)
library(dplyr)
library(gridExtra)

# Sample data
set.seed(123)
data <- data.frame(
  X = rnorm(100),
  Y = rnorm(100),
  Category = factor(sample(1:5, 100, replace = TRUE)),
  Size = runif(100, 1, 5),
  Value = rpois(100, lambda = 5)
)

# Scatterplot
scatterplot <- ggplot(data, aes(x = X, y = Y, color = Category)) +
  geom_point() +
  labs(title = "Scatterplot")

# Bubble Chart
bubble_chart <- ggplot(data, aes(x = X, y = Y, size = Size, color = Category)) +
  geom_point() +
  labs(title = "Bubble Chart")

# Bar Chart
bar_chart <- ggplot(data, aes(x = Category)) +
  geom_bar() +
  labs(title = "Bar Chart")

# Dot Plots
dot_plots <- ggplot(data, aes(x = Value)) +
  geom_histogram(binwidth = 0.5, fill = "blue", color = "black") +
  labs(title = "Dot Plots")

# Histogram
histogram <- ggplot(data, aes(x = X)) +
  geom_histogram(binwidth = 0.2, fill = "lightblue") +
  labs(title = "Histogram")

# Box Plot
box_plot <- ggplot(data, aes(x = Category, y = X, fill = Category)) +
  geom_boxplot() +
  labs(title = "Box Plot")

# Pie Chart
pie_chart_data <- data %>%
  group_by(Category) %>%
  summarize(Count = n())
pie_chart <- ggplot(pie_chart_data, aes(x = "", y = Count, fill = Category)) +
  geom_bar(stat = "identity") +
  coord_polar(theta = "y") +
  labs(title = "Pie Chart")

# Arrange and display the charts
grid.arrange(scatterplot, bubble_chart, bar_chart, dot_plots, histogram, box_plot, pie_chart, ncol = 2)