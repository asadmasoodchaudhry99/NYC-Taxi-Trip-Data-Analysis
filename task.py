import pandas as pd
import pyarrow.parquet as pq
import pyarrow 
import requests
from bs4 import BeautifulSoup
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from multiprocessing import Pool, Manager
from concurrent.futures import ThreadPoolExecutor,as_completed
import os
import tempfile

#Function to Download Parquet Files
def download_parquet_file(file_url):
    response = requests.get(file_url)
    file_content = response.content

    try:
        # Check if the file content starts with the Parquet magic bytes
        if file_content[:4] == b'PAR1':
            parquet_file = io.BytesIO(file_content)
            parquet_reader = pq.ParquetFile(parquet_file)
            df = parquet_reader.read(columns=['tpep_dropoff_datetime', 'tpep_pickup_datetime']).to_pandas()
            df['file_url'] = file_url
            df['month'] = file_url.split('_')[-1].split('.')[0]  # Add 'month' column
            print(f"Downloaded and processed file: {file_url}")
            return df
        else:
            print(f"Invalid or corrupted file: {file_url}")
            return None
    except pyarrow.ArrowInvalid:
        print(f"Skipping corrupt or invalid file: {file_url}")
        return None
    except Exception as e:
        print(f"Error downloading or processing file: {file_url}")
        print(str(e))
        return None

#Function to Write Data Frame to Parquet
def write_dataframe_to_parquet(df, output_dir, index):
    parquet_file = f"{output_dir}/part{index}.parquet"
    table = pyarrow.Table.from_pandas(df)
    pq.write_table(table, parquet_file, compression='snappy')
    print(f"Written part {index} to Parquet file: {parquet_file}")

#Function to Combine all parquet files
def CreateCombinedParquet():
    # Data Acquisition
    data_url = 'https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page'
    output_dir = 'ParquetFiles'
    output_file = 'Combined_TripData.parquet'
    batch_size = 3  # Number of DataFrames to process in each batch
    countfiles = 0
    # Create directory for Parquet files
    os.makedirs(output_dir, exist_ok=True)

    # Download the data file
    response = requests.get(data_url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    download_links = soup.find_all('a', href=True)
    downloaded_files = []
    for link in download_links:
        file_url = link['href']
        if file_url.endswith('.parquet') and 'yellow_tripdata' in file_url:
            downloaded_files.append(file_url)

    # Download Parquet files
    with ThreadPoolExecutor() as executor:
        dfs = executor.map(download_parquet_file, downloaded_files)
        dfs = [df for df in dfs if df is not None]

    # Write each batch of DataFrames to separate Parquet files and delete the DataFrames
    for i in range(0, len(dfs), batch_size):
        batch_dfs = dfs[i:i+batch_size]
        for j, df in enumerate(batch_dfs):
            write_dataframe_to_parquet(df, output_dir, i+j)
            del df  # Delete the DataFrame
            countfiles = countfiles + 1
            

# Combine Parquet files into a single file in batches if any files were written
    if countfiles > 0:
        parquet_files = [f"{output_dir}/part{i}.parquet" for i in range(0, countfiles, batch_size)]

        
        batch_index = 0

        for i in range(0, len(parquet_files), batch_size):
            batch_files = parquet_files[i:i+batch_size]
            combined_table = None

            for file in batch_files:
                table = pq.read_table(file)
                if combined_table is None:
                    combined_table = table
                else:
                    combined_table = pyarrow.concat_tables([combined_table, table])

            # Write the combined table for each batch
            if combined_table is not None:
                batch_output_file = f"{output_file}_part{batch_index}.parquet"
                pq.write_table(combined_table, batch_output_file, compression='snappy')
                print(f"Combined Parquet file saved as {batch_output_file}.")
                batch_index += 1

        if batch_index > 0:
            # Combine all batch files into a final single file
            final_output_file = f"{output_file}"
            final_parquet_files = [f"{output_file}_part{i}.parquet" for i in range(batch_index)]
            final_combined_table = None

            for file in final_parquet_files:
                table = pq.read_table(file)
                if final_combined_table is None:
                    final_combined_table = table
                else:
                    final_combined_table = pyarrow.concat_tables([final_combined_table, table])
                os.remove(file)

            if final_combined_table is not None:
                pq.write_table(final_combined_table, final_output_file, compression='snappy')
                print(f"Final combined Parquet file saved as {final_output_file}.")
                del final_combined_table  # Delete the final_combined_table after writing
        else:
            print("No valid Parquet files were processed.")
    else:
        print("No valid Parquet files were processed.")

#Function to calculate average trip length for the given month
def calculate_average_trip_length(file_path, month_name):
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(file_path)

    # Convert month name to numeric value
    month_dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                  'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    
    if not isinstance(month_name, str):
        raise ValueError("Invalid input type provided. Please enter a month name as a string.")

    month = month_dict.get(month_name.capitalize())
    
    if month is None:
        raise ValueError("Invalid month name provided. Please enter a valid month name.")

    # Filter the DataFrame for the specified month
    filtered_df = df[pd.to_datetime(df['month'], format='%Y-%m').dt.month == month]
   
    # Calculate the trip length in minutes
    trip_length = (filtered_df['tpep_dropoff_datetime'] - filtered_df['tpep_pickup_datetime']).dt.total_seconds()/60

    # Calculate the average trip length
    average_trip_length = trip_length.mean()

    return average_trip_length

#Function to plot average trip lenghts calculated for each month
def plot_average_trip_length_by_month(file_path):
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']

    average_trip_lengths = []
    for month in months:
        average_trip_length = calculate_average_trip_length(file_path, month)
        average_trip_lengths.append(average_trip_length)

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.bar(months, average_trip_lengths, color='skyblue')  # Set the bar color
    plt.xlabel('Month', fontsize=12)  # Set x-axis label and font size
    plt.ylabel('Average Trip Length (minutes)', fontsize=12)  # Set y-axis label and font size
    plt.title('Average Trip Length by Month', fontsize=14)  # Set title and font size
    plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels and set font size
    plt.yticks(fontsize=10)  # Set font size for y-axis ticks
    plt.grid(axis='y', linestyle='--')  # Add gridlines on the y-axis
    plt.tight_layout()  # Adjust the padding and spacing of the plot
    plt.show()

#Function to calculate rolling average of the trip length
def calculate_rolling_average_trip_length(file_path):
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(file_path)

    # Calculate the trip length in minutes
    df['trip_length'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

    # Convert the pickup datetime column to datetime format
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.date
    df['datetime_forYearly'] = pd.to_datetime(df['tpep_pickup_datetime'])
    
    # Create a unique identifier for each data point
    df.set_index('pickup_datetime', inplace=True)
    
    # Sort the DataFrame by date
    df.sort_index(inplace=True)
    
    # Calculate the rolling average trip length with a 45-day window
    df['Rolling_Average'] = df['trip_length'].rolling(window =45).mean()
    
    return df

#Function to plot rolling average by year
def plot_average_rolling_trip_length_by_year(df):
    # Group by year and calculate the average rolling average for each year
    df_yearly_avg = df.groupby(df['datetime_forYearly'].dt.year)['Rolling_Average'].mean()

    # Plot the average rolling trip length by year
    plt.plot(df_yearly_avg.index, df_yearly_avg.values)
    plt.xlabel('Year')
    plt.ylabel('Average Rolling Average')
    plt.title('Average Rolling Trip Length by Year')
    plt.xticks(df_yearly_avg.index)
    plt.show()





#RUN FUNCTIONS

file_path = 'Combined_TripData.parquet'

#STEP 1: Acquire relevant data from the link and combine it into one file.
'''
CreateCombinedParquet()
'''


#STEP 2: Calculate the average trip length for a month
'''
month_name = input("Enter the month name: ")
try:
    average_trip_length = round(calculate_average_trip_length(file_path, month_name), 2)
    print(f"Average trip length for {month_name}: {average_trip_length} Minutes")
except ValueError as e:
    print(str(e))
'''

#STEP 3: Plot the average trip length for each month
'''
plot_average_trip_length_by_month(file_path)
'''




#STEP 4: Calculate the Rolling Average
'''
combined_file_path = 'Combined_TripData.parquet'
df = calculate_rolling_average_trip_length(combined_file_path)
'''



#STEP 5: Plot the Rolling average once you have caculated the rolling average
'''
plot_average_rolling_trip_length_by_year(df)
'''

