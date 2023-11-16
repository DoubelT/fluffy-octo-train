file_path = r"D:\Uni\BA\ownModel\dataset\df_validationset_features"

with open(file_path, "r") as file:
    lines = file.readlines()

# Add square brackets at the end of each line
lines_with_brackets = ['[' + line.strip() + ']' for line in lines]

# Save the modified lines back to the file
output_file_path = r"D:\Uni\BA\ownModel\dataset\output_file.txt"
with open(output_file_path, "w") as output_file:
    output_file.write("\n".join(lines_with_brackets))

