import csv

# Prepare results file (assuming it already exists)
results_file = "boolq_results_20241222_124426.csv"


# CSV statistics analysis
print("Analyzing existing CSV results...")

# Initialize counters for statistics
total = 0
correct = 0
total_response_time = 0.0

# Read the CSV file and calculate statistics
with open(results_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        total_response_time += float(row['response_time'])
        if row['correct'].lower() == 'true':  # Check if the response was correct
            correct += 1

# Calculate metrics
accuracy = (correct / total) * 100 if total > 0 else 0
avg_response_time = total_response_time / total if total > 0 else 0

# Output the statistics
print(f"Total questions processed: {total}")
print(f"Correct answers: {correct}")
print(f"Incorrect answers: {total - correct}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Average response time: {avg_response_time:.2f} seconds")
