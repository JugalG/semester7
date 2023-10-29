# Parameters
size = 10  # Size of the Bloom filter's bit array
hash_functions = 3  # Number of hash functions to use

# Initialize the bit array as a list of zeros
bit_array = [0] * size

# Initialize a set to keep track of added items
added_items = set()

# Add items to the Bloom filter
def add(item):
    for i in range(hash_functions):
        index = hash(item + str(i)) % size
        bit_array[index] = 1
    added_items.add(item)

# Check for membership
def contains(item):
    for i in range(hash_functions):
        index = hash(item + str(i)) % size
        if bit_array[index] == 0:
            return False
    return item in added_items  # Check if the item is in the set of added items

# Add items to the Bloom filter
items_to_add = ["apple", "banana", "cherry"]
for item in items_to_add:
    add(item)

# Check for membership
items_to_check = ["apple", "banana", "cherry", "orange"]
for item in items_to_check:
    if contains(item):
        if item in items_to_add:
            print(f"'{item}' is a true positive")
        else:
            print(f"'{item}' is a false positive")
    else:
        print(f"'{item}' is not in the set")