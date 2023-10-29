# Load necessary libraries
library(ggplot2)
# Create the data frame
df <- data.frame(
  Subject = c(1, 2, 1, 2, 1, 2),
  Class = c(1, 2, 1, 2, 1, 2),
  Marks = c(56, 75, 48, 69, 84, 53)
)


# Create a subset where Subject < 3 and Class == 2
subset_df <- df[df$Subject < 3 & df$Class == 2, ]

# View the subset
print(subset_df)




# Create a scatterplot to visualize the relationship between Subject and Marks
ggplot(df, aes(x = Subject, y = Marks)) +
  geom_point() +
  labs(title = "Scatterplot of Subject vs. Marks", x = "Subject", y = "Marks")