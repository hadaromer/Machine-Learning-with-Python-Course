import os

# Function to check wheater a file exists, exit the program if not
def check_file_exists(results_filename):
    is_exist = os.path.exists(results_filename)
    if not is_exist:
        print(f'Error: Results file "{results_filename}" not found.')
        exit(1)