total_sum = 0
count = 0

# Open the file for reading
with open("a.txt", "r") as file:
    # Iterate through each line in the file
    for line in file:
        # Try to convert the line to a float (assuming the numbers are floating-point numbers)
        try:
            number = float(line.strip())
            # Add the number to the total sum
            total_sum += number
            # Increment the count
            count += 1
        except ValueError:
            # If a line doesn't represent a valid number, skip it
            continue

# Calculate the average
if count > 0:
    average = total_sum / count
    print(f"The average of the numbers in 'a.txt' is: {average:.2f}")
else:
    print("No valid numbers found in 'a.txt'")