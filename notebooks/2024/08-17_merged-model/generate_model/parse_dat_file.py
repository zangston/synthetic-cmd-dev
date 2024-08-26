def parse_dat_file(file_path):
    data = []

    # Open and read the file
    with open(file_path, 'r') as file:
        # Skip the first two lines (column names and units)
        file.readline()
        file.readline()

        # Read the rest of the file line by line
        for line in file:
            # Split each line into columns based on space(s)
            row = line.split()

            # Convert each value to the appropriate type if needed
            # Here we assume all are floats except the last column, which is a string
            row_data = [float(val) if i < len(row) - 1 else val for i, val in enumerate(row)]
            
            # Add the row to the data array
            data.append(row_data)

    return data
    