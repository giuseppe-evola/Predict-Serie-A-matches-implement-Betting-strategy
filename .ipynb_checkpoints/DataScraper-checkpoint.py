import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

def download_serie_a_data_by_season(start_season="1617", end_season="2425", output_folder="data"):
    """
    Downloads Serie A CSV files from the football-data.co.uk website for the given range of seasons
    and returns a dictionary of DataFrames, each corresponding to a season.
    
    Parameters:
        start_season (str): The first season to download (e.g., "1617" for 2016/2017).
        end_season (str): The last season to download (e.g., "2425" for 2024/2025).
        output_folder (str): The folder to save downloaded CSV files.
    
    Returns:
        dict: A dictionary where keys are seasons and values are DataFrames with the corresponding data.
    """
    # Base URLs
    base_url = "https://www.football-data.co.uk/italym.php"
    download_url_base = "https://www.football-data.co.uk/"
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Dictionary to store season DataFrames
    seasons_data = {}
    
    # Request the main page
    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Error requesting the main page: {response.status_code}")
        return None
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=True)
    
    for link in links:
        href = link['href']
        if "mmz4281" in href and "I1.csv" in href:
            # Extract season from the link
            season = href.split('/')[1]
            
            # Filter by season range
            if start_season <= season <= end_season:
                file_url = download_url_base + href
                file_name = os.path.join(output_folder, f"SerieA_{season}.csv")
                
                # Download the file if not already downloaded
                if not os.path.exists(file_name):
                    print(f"Downloading {file_url}...")
                    file_response = requests.get(file_url)
                    if file_response.status_code == 200:
                        with open(file_name, 'wb') as file:
                            file.write(file_response.content)
                        print(f"Saved: {file_name}")
                    else:
                        print(f"Error downloading {file_url}.")
                
                # Read the CSV into a DataFrame
                try:
                    season_data = pd.read_csv(file_name)
                    season_data['Season'] = season  # Add a season column
                    seasons_data[season] = season_data  # Store in dictionary with season as key
                except Exception as e:
                    print(f"Error reading {file_name}: {e}")
    
    # Return the dictionary with DataFrames
    if seasons_data:
        return seasons_data
    else:
        print("No data downloaded.")
        return None


