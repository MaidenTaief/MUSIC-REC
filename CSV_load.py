import pandas as pd

# csv file path
file_path = '/Users/taief/Desktop/MUSIC REC/data/data_by_artist.csv'

# load csv data into a DataFrame

data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(data.head())
