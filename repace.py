import re

csv_file_path = "caption_2025-05-21 02:14:48.486157.csv"

def replace_quotes(text):
    return re.sub(r'"', '', text)

def create_new_csv(input_file, output_file):
    with open(input_file    , 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Replace double quotes with empty string
            new_line = replace_quotes(line)
            outfile.write(new_line)

if __name__ == "__main__":
    # Create a new CSV file with replaced quotes
    create_new_csv(csv_file_path, "new_" + csv_file_path)
    print("CSV file created successfully.")
    

