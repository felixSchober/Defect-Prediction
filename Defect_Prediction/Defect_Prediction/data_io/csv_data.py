import csv

def get_csv_row_generator(path, delimiter=';', skip_first_row=True):
    with open(path) as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if skip_first_row:
                skip_first_row = False
                continue
            yield row
        


