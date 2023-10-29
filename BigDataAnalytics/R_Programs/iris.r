# Load necessary libraries
library(ggplot2)

# Load the Iris dataset from a CSV file
iris_data <- read.csv("/Users/surajchavan/Downloads/Iris.csv")

# View the first few rows of the dataset
head(iris_data)

# Scatterplot
scatterplot <- ggplot(iris_data, aes(x = SepalLengthCm, y = SepalWidthCm, color = Species)) +
  geom_point() +
  labs(title = "Scatterplot of Sepal Length vs. Sepal Width", x = "Sepal Length (cm)", y = "Sepal Width (cm)")

# Bubble Chart
bubble_chart <- ggplot(iris_data, aes(x = SepalLengthCm, y = SepalWidthCm, size = PetalLengthCm, color = Species)) +
  geom_point() +
  labs(title = "Bubble Chart of Sepal Length vs. Sepal Width", x = "Sepal Length (cm)", y = "Sepal Width (cm)")



# Dot Plots (Not typically used for this type of data)

# Histogram
histogram <- ggplot(iris_data, aes(x = SepalLengthCm, fill = Species)) +
  geom_histogram(binwidth = 0.2, alpha = 0.7) +
  labs(title = "Histogram of Sepal Length", x = "Sepal Length (cm)", y = "Frequency")

# Box Plot
box_plot <- ggplot(iris_data, aes(x = Species, y = SepalLengthCm, fill = Species)) +
  geom_boxplot() +
  labs(title = "Box Plot of Sepal Length by Species", x = "Species", y = "Sepal Length (cm)")

grid.arrange(scatterplot, bubble_chart, histogram, box_plot, ncol = 2)