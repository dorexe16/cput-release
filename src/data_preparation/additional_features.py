import os
import requests
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from googletrans import Translator
import folium
from datetime import datetime
from Data_Preparation import *
import os
import requests
import geopandas as gpd
from shapely.geometry import Point
from pyrosm import OSM
import json



file_path = "israel-latest.osm.pbf"

def download_israel_osm_data():
    url = "https://download.geofabrik.de/asia/israel-and-palestine-latest.osm.pbf"
    if not os.path.exists(file_path):
        response = requests.get(url, stream=True)
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

def prepare_neighborhood_index():
    # Load OSM data
    osm = OSM(file_path)
    gdf = osm.get_data_by_custom_criteria(
        custom_filter={"place": ["neighbourhood"]},
        osm_keys_to_keep=["name"]
    )
    # Build a spatial index for neighborhoods
    gdf = gdf.to_crs("EPSG:4326")  # Ensure coordinates are in lat/lon
    return gdf, gdf.sindex

def get_geographic_info_opencage(latitude, longitude):
    key = os.environ.get("OPENCAGE_API_KEY", "")  # Replace with your OpenCage API key
    geocoder = OpenCageGeocode(key)
    result = geocoder.reverse_geocode(latitude, longitude, no_annotations='1', language='en')
    
    if result:
        location = result[0]
        components = location['components']
#         components = location
#         return components
        return {
            "neighborhood": components.get('neighbourhood') or components.get('suburb'),
            "suburb": components.get('suburb'),
            "city": components.get('city') or components.get('town'),
            "state": components.get('state'),
            "country": components.get('country'),
            "postal_code": components.get('postcode'),
            "road": components.get('road')
        }
    else:
        return {"error": "Location not found"}

from opencage.geocoder import OpenCageGeocode
def get_neighborhood_name(lat, lon, gdf=None, spatial_index=None):
#     # Create a Point object
#     point = Point(lon, lat)
    
#     # Use spatial index to get possible matches
#     possible_matches_index = list(spatial_index.intersection(point.bounds))
#     possible_matches = gdf.iloc[possible_matches_index]
    
#     # Check each candidate polygon for containment
#     for idx, neighborhood in possible_matches.iterrows():
#         if neighborhood['geometry'].contains(point):
#             dict_obj = json.loads(neighborhood['tags'])
#             if 'name:en'  in dict_obj:
#                 return dict_obj['name:en']
#             else:
#                 print(lat, lon)
    lat = str(lat)+'01'
    lon = str(lon)+'01'
    dict1 = get_geographic_info_opencage(lat,lon)
    return dict1["neighborhood"] if dict1["neighborhood"] else dict1["road"]




# def get_neighborhood_name(latitude, longitude):
#     url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}"
    
#     try:
#         response = requests.get(url)
#         i=0
#         while response.status_code != 200:
#             response = requests.get(url)
#             i+=1
#             if i >=5:
#                 break
#         # Check if the request was successful
#         if response.status_code != 200:
# #             print( f"Error: Received status code {response.status_code}")
#             gdf, spatial_index = prepare_neighborhood_index()
#             return get_neighborhood_nameb(latitude, longitude, gdf, spatial_index)

        
#         # Parse the response JSON
#         data = response.json()
        
#         if 'address' in data:
#             if 'neighbourhood' in data['address']:
#                 return data['address']['neighbourhood']
#             elif 'suburb' in data['address']:
#                 return data['address']['suburb']
#             elif 'town' in data['address']:
#                 return data['address']['town']
#             elif 'city' in data['address']:
#                 return data['address']['city']
#             else:
#                 return "Neighborhood not found"
#         else:
#             return "Neighborhood not found"
    
#     except requests.exceptions.RequestException as e:
#         print(f"Request failed: {e}")
    
#     except ValueError as e:
#         print(f"JSON decoding failed: {e}")
        
from googletrans import Translator

def translate_hebrew_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='he', dest='en')
    return translation.text

def Prepare_metadata_by_vehicle_number(vehicle_ids = 460631):
    PATH_raw_data = '/data/inputs/Year_Data'
    vehicle_ids = [vehicle_ids]
    directory_path = f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{vehicle_ids}"

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    for year_str in ['2018','2019']:
        for month_index in tqdm(range(1, 13, 1)):
            if os.path.exists(f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{vehicle_ids}/{month_index}_{year_str}_trips.pickle")\
    and os.path.exists(f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{vehicle_ids}/{month_index}_{year_str}_car.csv"):
                continue
        
            months_metadata = pd.DataFrame(
                {'month_num_str': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
                 'month_shortcut_name': ['jan', 'feb', 'march', 'april', 'may', 'june', 'july', 'aug',
                                         'sep', 'oct', 'nov', 'dec'],
                 'days': [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]})
            if year_str == '2020':
                months_metadata['days'][1] = 29
            num_days = [int(np.floor((months_metadata.days[month_index - 1] - i) / 7 + 1)) for i in
                        range(1, 8, 1)]  # calculate number of days in each day of the week

            drives1, gps1, _ = read_same_trips('01', months_metadata.month_num_str[month_index - 1], year_str, num_days[0],
                                               directory=PATH_raw_data + '/')  # Saturdays of July
            drives1, gps1 = drives1.loc[drives1['VEHICLE_ID'].isin(vehicle_ids)], gps1.loc[gps1['vehicle_id'].isin(vehicle_ids)]
            drives1['START_DRIVE'] = pd.to_datetime(drives1['START_DRIVE'])
            drives1['END_DRIVE'] = pd.to_datetime(drives1['END_DRIVE'])
            gps1['orig_time'] = pd.to_datetime(gps1['orig_time'])

            drives2, gps2, _ = read_same_trips('02', months_metadata.month_num_str[month_index - 1], year_str, num_days[1],
                                               directory=PATH_raw_data + '/')  # Sundays of July
            drives2, gps2 = drives2.loc[drives2['VEHICLE_ID'].isin(vehicle_ids)], gps2.loc[gps2['vehicle_id'].isin(vehicle_ids)]
            drives2['START_DRIVE'] = pd.to_datetime(drives2['START_DRIVE'])
            drives2['END_DRIVE'] = pd.to_datetime(drives2['END_DRIVE'])
            gps2['orig_time'] = pd.to_datetime(gps2['orig_time'])

            drives3, gps3, _ = read_same_trips('03', months_metadata.month_num_str[month_index - 1], year_str, num_days[2],
                                               directory=PATH_raw_data + '/')  # Mondays of July
            drives3, gps3 = drives3.loc[drives3['VEHICLE_ID'].isin(vehicle_ids)], gps3.loc[gps3['vehicle_id'].isin(vehicle_ids)]
            drives3['START_DRIVE'] = pd.to_datetime(drives3['START_DRIVE'])
            drives3['END_DRIVE'] = pd.to_datetime(drives3['END_DRIVE'])
            gps3['orig_time'] = pd.to_datetime(gps3['orig_time'])

            drives4, gps4, _ = read_same_trips('04', months_metadata.month_num_str[month_index - 1], year_str, num_days[3],
                                               directory=PATH_raw_data + '/')  # Tuesdays of July
            drives4, gps4 = drives4.loc[drives4['VEHICLE_ID'].isin(vehicle_ids)], gps4.loc[gps4['vehicle_id'].isin(vehicle_ids)]
            drives4['START_DRIVE'] = pd.to_datetime(drives4['START_DRIVE'])
            drives4['END_DRIVE'] = pd.to_datetime(drives4['END_DRIVE'])
            gps4['orig_time'] = pd.to_datetime(gps4['orig_time'])

            drives5, gps5, _ = read_same_trips('05', months_metadata.month_num_str[month_index - 1], year_str, num_days[4],
                                               directory=PATH_raw_data + '/')  # Wendsdays of July
            drives5, gps5 = drives5.loc[drives5['VEHICLE_ID'].isin(vehicle_ids)], gps5.loc[gps5['vehicle_id'].isin(vehicle_ids)]
            drives5['START_DRIVE'] = pd.to_datetime(drives5['START_DRIVE'])
            drives5['END_DRIVE'] = pd.to_datetime(drives5['END_DRIVE'])
            gps5['orig_time'] = pd.to_datetime(gps5['orig_time'])

            drives6, gps6, _ = read_same_trips('06', months_metadata.month_num_str[month_index - 1], year_str, num_days[5],
                                               directory=PATH_raw_data + '/')  # Theursdays of July
            drives6, gps6 = drives6.loc[drives6['VEHICLE_ID'].isin(vehicle_ids)], gps6.loc[gps6['vehicle_id'].isin(vehicle_ids)]
            drives6['START_DRIVE'] = pd.to_datetime(drives6['START_DRIVE'])
            drives6['END_DRIVE'] = pd.to_datetime(drives6['END_DRIVE'])
            gps6['orig_time'] = pd.to_datetime(gps6['orig_time'])

            drives7, gps7, _ = read_same_trips('07', months_metadata.month_num_str[month_index - 1], year_str, num_days[6],
                                               directory=PATH_raw_data + '/')  # Fridays of July
            drives7, gps7 = drives7.loc[drives7['VEHICLE_ID'].isin(vehicle_ids)], gps7.loc[gps7['vehicle_id'].isin(vehicle_ids)]
            drives7['START_DRIVE'] = pd.to_datetime(drives7['START_DRIVE'])
            drives7['END_DRIVE'] = pd.to_datetime(drives7['END_DRIVE'])
            gps7['orig_time'] = pd.to_datetime(gps7['orig_time'])

            drives = pd.concat([drives1, drives2, drives3, drives4, drives5, drives6, drives7], ignore_index=True)
            gps = pd.concat([gps1, gps2, gps3, gps4, gps5, gps6, gps7], ignore_index=True)

            trip_len_min = 10
            drives_sub_group = drives.loc[drives['DRIVE_DURATION'] >= trip_len_min]
            
            trips_lst = separate_to_drives(drives_sub_group, gps)
            with open(
                    f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{vehicle_ids}/{month_index}_{year_str}_trips.pickle",
                    'wb') as handle:
                pickle.dump(trips_lst, handle)
            
            
            drives_sub_group.to_csv(
                f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{vehicle_ids}/{month_index}_{year_str}_car.csv",
                mode='w', index=False)
            
            
def concated_data_with_feat(car_id):  
    vehicle_ids = car_id # the needed ID - Irads car
    df_lst = []
    for year in ['2018','2019']:
        for i in range(1,13,1): 
            if len(str(i))==1:
                curr_df = export_processed_data_of_veich(month=str(i),year=year,car_id=vehicle_ids)
                curr_df.columns = curr_df.columns.str.lower()
                df_lst.append(curr_df)
            else:
                curr_df = export_processed_data_of_veich(month=str(i),year=year,car_id=vehicle_ids)
                curr_df.columns = curr_df.columns.str.lower()
                df_lst.append(curr_df)
    df = pd.concat(df_lst)
    df.columns = df.columns.str.upper()
    df.index = df['DRIVE_ID']
    return  df




def aggregating_car_data_for_pattern(car_id):
    path = f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{[car_id]}/agg_patterned_data.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        
        df = concated_data_with_feat(car_id)
        df.columns = df.columns.str.upper()
        # Extract the hour part from the "START_DRIVE" column and store it in a new column "START_HOUR"
        df["START_HOUR"] = pd.to_datetime(df["START_DRIVE"]).dt.hour

        # Extract the hour part from the "END_DRIVE" column and store it in a new column "END_HOUR"
        df["END_HOUR"] = pd.to_datetime(df["END_DRIVE"]).dt.hour

        # Extract the date (without time) from the "START_DRIVE" column and store it in a new column "DATE"
        df["DATE"] = pd.to_datetime(df["START_DRIVE"]).dt.date

        # Create a new column "DAY_OF_WEEK" that contains the day of the week (e.g., Monday, Tuesday)
        df["DAY_OF_WEEK"] = df["DATE"].apply(lambda x: x.strftime("%A"))

        # Set the index of the DataFrame to the values of the "DRIVE_ID" column
        df.index = df['DRIVE_ID']

        # Drop the "START_DRIVE" and "END_DRIVE" columns as they are no longer needed
        df = df.drop(["START_DRIVE", "END_DRIVE"], axis=1)
        grouped_df = df.copy()

        # Round the "START_LATITUDE" column from `grouped_df` to 3 decimal places and assign it to `grouped_df`
        grouped_df["START_LATITUDE"] = grouped_df["START_LATITUDE"].round(3)

        # Round the "START_LONGITUDE" column from `grouped_df` to 3 decimal places and assign it to `grouped_df`
        grouped_df["START_LONGITUDE"] = grouped_df["START_LONGITUDE"].round(3)

        # Round the "END_LATITUDE" column from `grouped_df` to 3 decimal places and assign it to `grouped_df`
        grouped_df["END_LATITUDE"] = grouped_df["END_LATITUDE"].round(3)

        # Round the "END_LONGITUDE" column from `grouped_df` to 3 decimal places and assign it to `grouped_df`
        grouped_df["END_LONGITUDE"] = grouped_df["END_LONGITUDE"].round(3)

        # Group the DataFrame by the rounded "END_LATITUDE" and "END_LONGITUDE" columns
        # Count the number of occurrences (drives) for each unique pair of latitude and longitude
        # Reset the index and store the result in a new column named 'num_drives' 
        grouped_df = grouped_df.groupby(['END_LATITUDE', 'END_LONGITUDE']).size().reset_index(name='num_drives')

        most_driven_cor = grouped_df.sort_values(by='num_drives', ascending=False).iloc[0]
        # Extract latitude and longitude from the selected row
        latitude = most_driven_cor['END_LATITUDE']
        longitude = most_driven_cor['END_LONGITUDE']

        # Create a tuple of the most driven coordinates
        home_cor = (latitude, longitude)

        agg_drives_filt = grouped_df[(grouped_df['num_drives'] > 8)&((grouped_df['END_LATITUDE']!=latitude) & (grouped_df['END_LONGITUDE']!=longitude))].sort_values(by=[ 'num_drives'], ascending=[ False])
        gdf, spatial_index = prepare_neighborhood_index()
        # agg_drives_filt = agg_drives_filt[(agg_drives_filt['START_LATITUDE'] != agg_drives_filt['END_LATITUDE'])&(agg_drives_filt['START_LONGITUDE'] != agg_drives_filt['END_LONGITUDE'])]
        agg_drives_filt['NEIGHBORHOOD'] = agg_drives_filt.apply(lambda row: get_neighborhood_name(row['END_LATITUDE'], row['END_LONGITUDE'],gdf, spatial_index), axis=1)
        agg_drives_filt.to_csv(path, index=False)
    return agg_drives_filt


def tag_drives(agg_drives,driver=1):
    drives_df = agg_drives.copy()
    drives_df['driver'] = 0
    driver_dict = {}
#     agg_drives.at[3263176, 'driver'] = 0
    for i, row in drives_df.iterrows():
#         agg_drives.at[i, 'driver'] = int(0)
        if  i == drives_df.index[0]:
            drives_df.at[i, 'driver'] = driver
            driver_dict[i] = row
        else:
            for j in driver_dict:
                if (
                    row['NEIGHBORHOOD'] == driver_dict[j]['NEIGHBORHOOD']):
#                     and row['START_HOUR'] == driver_dict[j]['START_HOUR']):
#                     and row['END_LATITUDE'] == driver_dict[j]['END_LATITUDE']
#                     and row['END_LONGITUDE'] == driver_dict[j]['END_LONGITUDE'] ):

                        drives_df.at[i, 'driver'] = driver
                        break  # Exit the loop once a match is found
    return drives_df[drives_df['driver']==1] , drives_df[drives_df['driver']==0]

def check_dicts_exist(path):
    # Define the base path using car_id
    
    
    # Define file paths for each dictionary
    path_dict_sum = os.path.join(path, 'dict_sum.pkl')
    path_dict_cor = os.path.join(path, 'dict_cor.pkl')
    path_dict_drives = os.path.join(path, 'dict_drives.pkl')
    
    # Check if each file exists and store the results in a dictionary
    files_exist = {
        'dict_sum': os.path.exists(path_dict_sum),
        'dict_cor': os.path.exists(path_dict_cor),
        'dict_drives': os.path.exists(path_dict_drives)
    }
    files_exist = {
        'dict_sum': False,
        'dict_cor': False,
        'dict_drives': False
    }
    return files_exist


def get_drives(df, drives_group, with_returns=True):   
    # List to store indices of matching drives
    matching_drives = []

    # Iterate over each row in the dataframe and drives_group
    for index_df, row_df in df.iterrows():
        for index_dg, row_dg in drives_group.iterrows():
            # Check if the end coordinates of the current drive in df match
            # the end coordinates of the current drive in drives_group
            if (round(row_df['END_LATITUDE'], 3) == row_dg['END_LATITUDE'] and
                round(row_df['END_LONGITUDE'], 3) == row_dg['END_LONGITUDE']):
                # If match found, append index of the drive in df to matching_drives
                matching_drives.append(index_df)
                # Break the inner loop as we've found a match
                break

    if with_returns:
        # Iterate over the filtered dataframe (df) based on matching indices
        for index_df, row_df in df[df.index.isin(matching_drives)].iterrows():
            # Iterate again over the entire dataframe to find drives starting where
            # the matching drives end and on the same date
            for index, row in df.iterrows():
                if (round(row_df['END_LATITUDE'], 3) == round(row['START_LATITUDE'], 3) and
                    round(row_df['END_LONGITUDE'], 3) == round(row['START_LONGITUDE'], 3) and
                    row_df['DATE'] == row['DATE']):
                    # If conditions match, append index of the drive in df to following_drives
                    matching_drives.append(index)
                    break
    
    return matching_drives

def dict_creation_driving_groups(agg_drives_filt,car_id):
    dict_drives = {}
    dict_sum = {}
    dict_cor = {}
    path = f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{[car_id]}"
    checker = check_dicts_exist(path)
    if not checker['dict_sum'] and not checker['dict_cor'] and not checker['dict_drives']:
    
        # Make a copy of the filtered drives DataFrame (agg_drives_filt) to preserve the original data
        agg_drives2 = agg_drives_filt.copy()
        df = concated_data_with_feat(car_id)
        df.columns = df.columns.str.upper()
        df["DATE"] = pd.to_datetime(df["START_DRIVE"]).dt.date
        # Initialize a counter `i` to label different driver groups
        i = 1

        # Loop until there is only one drive left in the agg_drives2 DataFrame
        while len(agg_drives2) > 1:

            # Tag the drives in the remaining DataFrame (agg_drives2), separating the current group of drives
            # `tag_drives_group`: the drives tagged with the current driver (1)
            # `agg_drives2`: the remaining drives not yet tagged (driver 0)
            tag_drives_group, agg_drives2 = tag_drives(agg_drives2)

            # Get a list of drive indices from the original DataFrame (df) that match the coordinates of the tagged group
            driver1_lst = get_drives(df, tag_drives_group)

            # Create a new DataFrame (driver1_df) that contains only the matching drives from the cleaned DataFrame
            # - Filter columns to keep only the first 15 columns from `columns_og` along with 'END_LATITUDE' and 'END_LONGITUDE'
            driver1_df = df[df.index.isin(driver1_lst)][ ['END_LATITUDE', 'END_LONGITUDE']]

            # Calculate the total number of drives (rows) in this group
            sum_drives = len(driver1_df)

            # If the group contains any drives, proceed to record data
            if sum_drives > 0:

                # Store the coordinates (latitude, longitude) of the first drive in the group
                dict_cor[i] = (float(tag_drives_group.head(1)['END_LATITUDE']), float(tag_drives_group.head(1)['END_LONGITUDE']))

                # Store the count of drives in the current group
                dict_sum[i] = sum_drives

                # Store the DataFrame of drives for the current group
                dict_drives[i] = driver1_df

            # Increment the counter `i` for the next group
            i += 1

        # Sort the `dict_sum` dictionary by the number of drives in descending order
        # - This step ensures that the groups with the largest number of drives come first
        dict_sum = {k: v for k, v in sorted(dict_sum.items(), key=lambda item: item[1], reverse=True)}

        # Reorder `dict_cor` based on the sorted keys of `dict_sum`, so that the coordinates are consistent with the sorted groups
        dict_cor = {k: dict_cor[k] for k in dict_sum.keys()}

        # Reorder `dict_drives` similarly, to match the sorted group order
        dict_drives = {k: dict_drives[k] for k in dict_sum.keys()}
        # Define file paths for each dictionary
        path_dict_sum = os.path.join(path, 'dict_sum.pkl')
        path_dict_cor = os.path.join(path, 'dict_cor.pkl')
        path_dict_drives = os.path.join(path, 'dict_drives.pkl')

        # Save each dictionary as a pickle file
        with open(path_dict_sum, 'wb') as file:
            pickle.dump(dict_sum, file)

        with open(path_dict_cor, 'wb') as file:
            pickle.dump(dict_cor, file)

        with open(path_dict_drives, 'wb') as file:
            pickle.dump(dict_drives, file)
    else: 
        path_dict_sum = os.path.join(path, 'dict_sum.pkl')
        path_dict_cor = os.path.join(path, 'dict_cor.pkl')
        path_dict_drives = os.path.join(path, 'dict_drives.pkl')
        with open(path_dict_sum, 'rb') as file:
            dict_sum = pickle.load(file)
        with open(path_dict_cor, 'rb') as file:
            dict_cor = pickle.load(file)
        with open(path_dict_drives, 'rb') as file:
            dict_drives = pickle.load(file)
    return dict_sum, dict_cor, dict_drives


def remap_keys(dicts):
    # Find the minimum key in all dictionaries
    min_key = min(min(d) for d in dicts)
    # Create a mapping from old keys to new keys
    key_mapping = {old_key: idx + min_key for idx, old_key in enumerate(sorted(set.union(*map(set, dicts))))}
    # Remap keys in all dictionaries
    remapped_dicts = [{key_mapping[key]: value for key, value in d.items()} for d in dicts]
    return tuple(remapped_dicts)


def fixing_and_sorting_dicts(dict_sum, dict_cor, dict_drives):
    keys_to_delete = [key for key, value in dict_sum.items() if value < 60]
    for key in keys_to_delete:
        del dict_sum[key]
        del dict_drives[key]
        del dict_cor[key]
    lst = remap_keys([dict_sum, dict_drives, dict_cor])
    dict_sum =lst[0]
    dict_drives = lst[1]
    dict_cor = lst[2]
    dict_sum = {k: v for k, v in sorted(dict_sum.items(), key=lambda item: item[1], reverse=True)}

    dict_cor = {k: dict_cor[k] for k in dict_sum.keys()}
    dict_drives = {k: dict_drives[k] for k in dict_sum.keys()}
    neigh_dict = {}
    for group in dict_cor:
        neigh_dict[group] = get_neighborhood_name(dict_cor[group][0], dict_cor[group][1])
        print("Neighborhood:", neigh_dict[group],group)
    return dict_sum, dict_cor, dict_drives, neigh_dict


def acceleration_estimation(trip_record):
    """
    Estimate the acceleration with the speed data.
    :param trip_record: data frame with the original GPS meta-data
    :return: trip_record data frame with new columns - prev_time, next_time, prev_speed, next_speed,
                acceleration_est_1/2/3
    """
    trip_record['orig_time'] = pd.to_datetime(trip_record['orig_time'])
    trip_record['prev_time'] = trip_record['orig_time'].shift(1)
    trip_record['next_time'] = trip_record['orig_time'].shift(-1)
    trip_record['prev_time'] = pd.to_datetime(trip_record['prev_time'])
    trip_record['next_time'] = pd.to_datetime(trip_record['next_time'])
    trip_record['prev_speed'] = trip_record['speed'].shift(1)
    trip_record['next_speed'] = trip_record['speed'].shift(-1)
    trip_record = trip_record.copy().loc[~(trip_record['next_time'] == trip_record['prev_time'])]
    trip_record['acceleration_est_1'] = trip_record.apply(lambda row: (row['next_speed'] - row['prev_speed']) /
                                                                      (row['next_time'] - row['prev_time']).seconds
                                                          , axis=1)
    trip_record.at[trip_record.index[0], 'acceleration_est_1'] = (trip_record['next_speed'].iloc[0] -
                                                                  trip_record['speed'].iloc[0]) / \
                                                                 (trip_record['next_time'].iloc[0] -
                                                                  trip_record['orig_time'].iloc[0]).seconds
    trip_record.at[trip_record.index[-1], 'acceleration_est_1'] = (trip_record['speed'].iloc[-1] -
                                                                   trip_record['prev_speed'].iloc[-1]) / \
                                                                  (trip_record['orig_time'].iloc[-1]
                                                                   - trip_record['prev_time'].iloc[-1]).seconds
    """trip_record['acceleration_est_2'] = trip_record.apply(lambda row: (row['speed'] - row['prev_speed']) /
                                                                      (row['orig_time'] - row['prev_time']).seconds,
                                                          axis=1)
    trip_record['acceleration_est_3'] = trip_record.apply(lambda row: (row['next_speed'] - row['speed']) /
                                                                      (row['next_time'] - row['orig_time']).seconds
                                                          , axis=1)"""
    return trip_record


def angular_acc_estimation(trip_record):
    """
    create the angular acc estimation
    :param trip_record: data frame having the shifted columns from the acceleration estimation function.
    :return: 3 new columns to the trip data frame - prev direction, next direction, and angular acc.
    """
    # trip_no_idle = trip_record.copy()
    trip_record = trip_record.copy().loc[~((trip_record['orig_time'] == trip_record['prev_time']) |
                                           (trip_record['orig_time'] == trip_record['next_time']))]
    trip_record['prev_direction'] = trip_record.direction.shift(1)
    trip_record['next_direction'] = trip_record.direction.shift(-1)
    # trip_record['prev_time'] = trip_record.orig_time.shift(1)
    # trip_record['next_time'] = trip_record.orig_time.shift(-1)
    trip_record = trip_record.copy().loc[~(trip_record['next_time'] == trip_record['prev_time'])]
    
    
    def calculate_angular_acc(row):
        try:
            return (row['next_direction'] - 2 * row['direction'] + row['prev_direction']) / \
                   ((row['next_time'] - row['orig_time']).total_seconds() * 
                    (row['orig_time'] - row['prev_time']).total_seconds())
        except ZeroDivisionError:
            return 0  # Assign 0 for angular acceleration in case of division by zero
    # Calculate angular acceleration
    trip_record['angular_acc'] = trip_record.apply(calculate_angular_acc, axis=1)
    
    return trip_record

def trip_on_map(df):
    # Create a map object
    m = folium.Map(location=[0, 0], zoom_start=2)

    # Define colors for markers
    colors = {2: 'green', 0: 'blue', 1: 'red', 3: 'orange', 4: 'pink', 5: 'teal', 6: 'maroon', 7: 'lavender'}

    color = colors[0]
    j = 0 


    for i, row in df.iterrows():
        if row['speed'] == 0:
            color = colors[1]
            rad = 8
        else:
            color = colors[0]
            rad = 5
        lat, lon = row['latitude'], row['longitude']
        folium.CircleMarker(location=[lat, lon], radius=rad, color=color, fill=True, fill_color=color).add_to(m)

    # Display the map
    return m

import folium

def create_map(dict_cor):
    # Create a base map
    m = folium.Map(location=[0, 0], zoom_start=2)

    # Define colors for markers
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'gray', 'black', 'pink', 'yellow', 'cyan', 'magenta', 'teal', 'maroon', 'navy', 'olive', 'lavender', 'peach', 'turquoise', 'silver', 'gold']

    j= 0 
    # Iterate through the list of data frames and add markers to the map
    for group in dict_cor:
        lat, lon = dict_cor[group]
        color = colors[j % len(colors)]
        j+=1# Select color based on index



        folium.CircleMarker(location=[lat, lon], radius=10, color=color, fill=True, fill_color=color).add_to(m)
    return m

def loading_ts_drives(car_id):
    path = f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{[car_id]}/ts_all_drives_lst.pickle"
    if os.path.exists(path):
        with open(path, 'rb') as file:
            ts_all_drives_lst = pickle.load(file)
        return ts_all_drives_lst
    else:
        all_trips = []

        for year in ['2018','2019']:
                for month in range(1,13,1): 

                    with open(f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{[car_id]}/{month}_{year}_trips.pickle", 'rb') as handle:
                            trips_lst = pickle.load(handle)
                    if len(trips_lst) >= 1:
                        for trip in trips_lst[0]:
                            try:
                                trip.time_series_record.drop_duplicates(subset=['orig_time'], keep=False)
                #                 curr_trip = speed_estimation(trip.time_series_record)
                                curr_trip = acceleration_estimation(trip.time_series_record)
                                curr_trip =angular_acc_estimation(curr_trip)
                                curr_trip = curr_trip[(curr_trip['vehicle_state'] == 1) | (curr_trip['vehicle_state'] == 2)]
                                curr_trip['date'] = curr_trip['orig_time'].dt.date
                                curr_trip['hour'] = curr_trip['orig_time'].dt.time
                                curr_trip['drive_id'] = trip.id

                                curr_trip.drop(['prev_time',
                                       'next_time','prev_speed','next_speed','prev_direction',
                                      'next_direction','mileage','vehicle_id'],axis=1,inplace=True)
                                curr_trip.fillna(0, inplace=True)
                                all_trips.append(curr_trip)
                            except:
                                continue

        with open(path, 'wb') as file:
            pickle.dump(all_trips, file)
                    
    return all_trips

def split_dataframe(df):
    # Initialize an empty list to store lists of indices that will represent individual DataFrames
    lst = []
    
    # Initialize an empty list to collect indices of the current drive segment
    df_indices = []

    # Iterate over each row of the DataFrame
    for i, row in df.iterrows():
        try:
            # Check if the current row's speed is 0 and the next row's speed is greater than 0
            # This condition is used to detect a transition from stationary to movement
            if row.speed == 0 and df.loc[i + 1].speed > 0:
                # If the condition is met, append the collected indices of the current segment to lst
                lst.append(df_indices)
                
                # Reset df_indices to start collecting indices for the next segment
                df_indices = []
        
        # Handle the case where there is no next row (to avoid KeyError on the last row)
        except KeyError:
            pass
        
        # Append the current index (i) to df_indices, as part of the ongoing segment
        df_indices.append(i)

    # After the loop, append the last segment's indices to lst (since the loop ends without appending)
    lst.append(df_indices)

    # Create a list of DataFrames by splitting the original DataFrame based on the collected indices
    # Each set of indices in lst is used to filter rows from the original DataFrame
    # Exclude any empty lists of indices (if any)
    dataframes = [df[df.index.isin(indices)] for indices in lst if indices]

    # Return the list of DataFrames
    return dataframes

def create_dict_groups2ts(dict_drives, ts_drives):
    # Initialize an empty dictionary to store the time series grouped by the group number
    groups_ts = {}

    # Iterate over each group in the dict_drives dictionary
    # 'group' represents the key (group number) in dict_drives, which corresponds to a specific set of drives
    for group in dict_drives:

        # For each group, iterate over each row (drive) in the DataFrame corresponding to that group
        # 'ind' represents the index of the row, and 'row' represents the row data itself
        for ind, row in dict_drives[group].iterrows():

            # Iterate over each time series in the ts_drives list (which contains filtered time series drives)
            for series in ts_drives:

                # Check if the mean of the 'drive_id' in the time series matches the index (ind) from the group
                # This is used to associate the time series with the group if the 'drive_id' matches
                if np.mean(series.drive_id) == ind:

                    # Get the list of time series for the current group from the dictionary
                    # If the group doesn't exist in groups_ts yet, initialize it as an empty list
                    groups_ts[group] = groups_ts.get(group, [])

                    # Append the matching time series to the group's list in the groups_ts dictionary
                    groups_ts[group].append(series)

                    # Break the inner loop to stop checking other series once a match is found for the current 'ind'
                    break
    return groups_ts

def create_list_spiltted_ts(ts_drives):
    splitted_ts_drives = []

# Iterate over each trip in the ts_drives list
    for trip in ts_drives:
        # Use the split_dataframe function to split the current trip into sub-trips
        sub_trips_lst = split_dataframe(trip)

        # Iterate over each sub-trip generated from the split
        for sub_trip in sub_trips_lst:
            # Append each sub-trip to the list of splitted_ts_drives
            splitted_ts_drives.append(sub_trip)
            
            
def create_feat_(groups_ts):
    # Initialize a dictionary to hold the processed time series data for each group
    group_ts_for_dbscan = {}

    # Iterate over each group in groups_ts
    for group in groups_ts:
        # Ensure the group key exists in group_ts_for_dbscan, initializing with an empty list if not
        group_ts_for_dbscan[group] = group_ts_for_dbscan.get(group, [])

        # Iterate over each series (trip) in the current group
        for series in groups_ts[group]:
            # Identify changes in latitude and longitude by comparing the current and previous row values
            series['lat_change'] = series['latitude'].shift() != series['latitude']
            series['lon_change'] = series['longitude'].shift() != series['longitude']

            # Combine the change indicators into a single column
            series['change'] = series['lat_change'] | series['lon_change']

            # Shift the change indicator to mark the last unchanged row
            series['keep'] = series['change'].shift(-1, fill_value=True)

            # Keep only the rows that are marked for keeping (unchanged until the next change)
            series = series[series['keep']].copy()

            # Drop the helper columns used for calculations, as they are no longer needed
            series.drop(columns=['lat_change', 'lon_change', 'change', 'keep'], inplace=True)

            # Recompute features on the deduplicated segment and keep the filtered result.
            series = acceleration_estimation(series)
            series = angular_acc_estimation(series)

            # Drop unnecessary columns that are not needed for DBSCAN processing
            series.drop(columns=['vehicle_state', 'date', 'direction', 'prev_time', 'next_time', 'prev_speed', 'next_speed', 'prev_direction', 'next_direction'], inplace=True)

            # Remove any rows with missing values to ensure clean data for clustering
            series.dropna(inplace=True)

            # Append the processed series to the corresponding group in group_ts_for_dbscan
            group_ts_for_dbscan[group].append(series)
    # Initialize a dictionary to hold the split time series data for each group
    splitted_ts_groups = {}

    # Iterate over each group in the dictionary containing time series data for DBSCAN
    for group in group_ts_for_dbscan:
        # Iterate over each DataFrame (time series) in the current group
        for curr_df in group_ts_for_dbscan[group]:
            # Ensure the current group key exists in splitted_ts_groups, initializing with an empty list if not
            splitted_ts_groups[group] = splitted_ts_groups.get(group, [])

            # Split the current DataFrame into smaller segments based on defined criteria in the split_dataframe function
            # and append the resulting segments to the list for the current group
            splitted_ts_groups[group] = splitted_ts_groups[group] + split_dataframe(curr_df)
    return splitted_ts_groups

def normalize_time_series(df_list):
    # Ensure all DataFrames in the list have the same shape by resetting their indices
    for df in df_list:
        df.reset_index(inplace=True, drop=True)

    # Concatenate all DataFrames along a new axis (rows)
    concatenated = pd.concat(df_list, axis=0)
    
    # Specify the columns of interest for normalization
    columns_of_interest = ['speed', 'acceleration_est_1', 'angular_acc']

    # Group by index and calculate the mean and standard deviation for the specified columns
    grouped = concatenated[columns_of_interest].groupby(concatenated.index)
    mean_series = grouped.mean()  # Calculate mean for each index
    std_series = grouped.std()    # Calculate standard deviation for each index
    
    # Replace any zero standard deviations with one to avoid division by zero during normalization
    std_series = std_series.replace(0, 1)

    # Initialize a list to hold the normalized DataFrames
    normalized_list = []
    
    # Normalize each DataFrame in the original list according to the calculated mean and standard deviation
    for df in df_list:
        normalized_df = df.copy()  # Create a copy of the current DataFrame
        index_values = normalized_df.index  # Get the index values of the current DataFrame
        
        # Normalize each column of interest based on its index value
        for col in columns_of_interest:
            # Apply normalization: (value - mean) / std_dev
            normalized_df[col] = (normalized_df[col] - mean_series.loc[index_values, col]) / std_series.loc[index_values, col]
        
        # Append the normalized DataFrame to the list
        normalized_list.append(normalized_df)
    
    # Return the list of normalized DataFrames
    return normalized_list


drives_hours = {1:[7,8,9,10,19,20,21,22,23],2:[11,12,13,14,15,16,17,18],3:[0,1,2,3,4,5,6]}

def scaling(splitted_ts_groups):
    scaling_parameters = {}  # Initialize an empty dictionary to store scaling parameters

# Iterate over each group in the splitted time series groups
    for group in splitted_ts_groups:
        # Iterate over each series in the current group
        for series in splitted_ts_groups[group]:
            series['group'] = group  # Assign the current group to the 'group' column in the series
            length = len(series)  # Get the length of the current series

            # Check if the length of the series is between 5 and 40
            if 5 < length < 40:
                road_speed = series.iloc[0]['road_speed']  # Get the road speed from the first row of the series

                # Determine the hour group based on the hour of the first entry in the series
                for h in drives_hours:
                    if int(series.iloc[0]['hour'].hour) in drives_hours[h]:
                        group_hour = h  # Assign the hour group based on the drives_hours

                # Create a tuple of parameters: (length, road_speed, group_hour)
                scaling_parameters[(length, road_speed, group_hour)] = scaling_parameters.get((length, road_speed, group_hour), [])
                # Append the current series to the list of series corresponding to the parameters
                scaling_parameters[(length, road_speed, group_hour)].append(series)
    return {key: scaling_parameters[key]  for key in scaling_parameters if len(scaling_parameters[key])>30}


from sktime.transformations.panel.rocket import MiniRocketMultivariate

# Define the convert_dfs_to_arrays function
def convert_dfs_to_arrays(df_list):
    arrays = []
    labels = []
    for df in df_list:
        arrays.append(df[['speed', 'acceleration_est_1', 'angular_acc']].values.T)  # Transpose to get shape (num_features, num_timesteps)
        labels.append(df.iloc[0]['group'])
    return np.stack(arrays, axis=0), labels

def compute_cluster(rand=100,neighbors_dict={},dict_length={}):
    """
    Computes clusters using MiniRocket transformation, PCA, and various clustering algorithms.
    Returns the labels for K-Means, GMM, DBSCAN, and Spectral Clustering.
    """
    kmeans_matrix = np.zeros((len(neighbors_dict), len(neighbors_dict)))
    gmm_matrix = np.zeros((len(neighbors_dict), len(neighbors_dict)))
    dbscan_matrix = np.zeros((len(neighbors_dict), len(neighbors_dict)))
    spectral_matrix = np.zeros((len(neighbors_dict), len(neighbors_dict)))
    def convert_dfs_to_arrays(df_list):
        """Convert list of DataFrames to numpy arrays."""
        arrays = []
        labels = []
        for df in df_list:
            arrays.append(df[['speed', 'acceleration_est_1', 'angular_acc']].values.T)  # Shape: (num_features, num_timesteps)
            labels.append(df.iloc[0]['group'])
        return np.stack(arrays, axis=0), labels

    t_labels = []  # True labels
    X_transformed_list = []  # List to store transformed arrays

    # Transform each DataFrame in dict_length using MiniRocket
    for key in dict_length:
        arrays, true_labels = convert_dfs_to_arrays(dict_length[key])
        t_labels += true_labels
        
        # Create and fit the MiniRocket transformer
        rocket = MiniRocketMultivariate(num_kernels=10000, random_state=int(rand))
        rocket.fit(arrays)

        # Transform the data and store it
        X_transformed = rocket.transform(arrays)
        X_transformed_list.append(X_transformed)

    # Concatenate all transformed arrays along the first axis
    X_transformed_all = np.concatenate(X_transformed_list, axis=0)

    # Apply PCA to reduce dimensions to 3 components
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_transformed_all)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca['group'] = t_labels

    # Calculate statistics for the largest cluster
    group_stats = calculate_stats_for_largest_cluster(df_pca)

    # Extract means for clustering
    X_means = np.array([stats['mean'] for stats in group_stats.values()])
    group_labels = list(group_stats.keys())

    def evaluate_clusterings(X, max_k=5):
        """Evaluate silhouette scores for K-Means and GMM."""
        silhouette_scores_kmeans = []
        silhouette_scores_gmm = []
        silhouette_scores_spectral = []

        for k in range(2, max_k + 1):
            # K-Means
            kmeans = KMeans(n_clusters=k)
            kmeans_labels = kmeans.fit_predict(X)
            silhouette_scores_kmeans.append(silhouette_score(X, kmeans_labels))

            # Gaussian Mixture Model
            gmm = GaussianMixture(n_components=k)
            gmm_labels = gmm.fit_predict(X)
            silhouette_scores_gmm.append(silhouette_score(X, gmm_labels))

            # Spectral Clustering
            spectral = SpectralClustering(n_clusters=k)
            spectral_labels = spectral.fit_predict(X)
            silhouette_scores_spectral.append(silhouette_score(X, spectral_labels))

        return silhouette_scores_kmeans, silhouette_scores_gmm, silhouette_scores_spectral

    # Evaluate silhouette scores for each clustering method
    silhouette_scores_kmeans, silhouette_scores_gmm, silhouette_scores_spectral = evaluate_clusterings(X_means)

    # Determine the best k for each clustering method based on silhouette scores
    best_k_kmeans = np.argmax(silhouette_scores_kmeans) + 2
    best_k_gmm = np.argmax(silhouette_scores_gmm) + 2
    best_k_spectral = np.argmax(silhouette_scores_spectral) + 2

    # Apply K-Means, GMM, DBSCAN, and Spectral Clustering with the best k
    kmeans_labels = KMeans(n_clusters=best_k_kmeans, random_state=42).fit_predict(X_means)
    gmm_labels = GaussianMixture(n_components=best_k_gmm, random_state=42).fit_predict(X_means)
    dbscan_labels = DBSCAN(eps=1.5, min_samples=1).fit_predict(X_means)
    spectral_labels = SpectralClustering(n_clusters=best_k_spectral, random_state=42).fit_predict(X_means)

    return kmeans_labels, gmm_labels, dbscan_labels, spectral_labels