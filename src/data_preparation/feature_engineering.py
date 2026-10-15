# TripGBDT Pre-processing according to article

# 57 Features described in word.

from Library import *
from tqdm import tqdm
import numpy as np

PATH_raw_data = '/data/inputs/Year_Data'
class Trip:
    def __init__(self, trip_id, trip_record):
        self.id = trip_id
        self.time_series_record = trip_record


def find_trip_obj(trip_id, trips_lst):
    for vehicle_i in range(len(trips_lst)):
        for trip in trips_lst[vehicle_i]:
            if trip.id == trip_id:
                return trip



def calc_angle_3_gps_points(point_1, point_2, point_3):
    """
    Calculate the moving angle of vehicle based on a window of three consecutive points.
    :param point_1: GPS coordinate record
    :param point_2: GPS coordinate record
    :param point_3: GPS coordinate record
    :return: angle between 0 to 180 degrees.
    """
    ac = point_2 - point_1
    cb = point_2 - point_3
    cosine_angle = np.dot(ac, cb) / (np.linalg.norm(ac) * np.linalg.norm(cb))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def calc_moving_angle_for_trip(trip_record):
    """
    Calculate the moving degree for trip's points
    :param trip_record: data frame
    :return: the data frame with additional cols - prev_lat+lon, next_lat+lon, moving angle (Nan for first, last points
    or no moving)
    """
    trip_record['prev_lat'] = trip_record['latitude'].shift(1)
    trip_record['prev_lon'] = trip_record['longitude'].shift(1)
    trip_record['next_lat'] = trip_record['latitude'].shift(-1)
    trip_record['next_lon'] = trip_record['longitude'].shift(-1)
    trip_record['moving_angle'] = trip_record.apply(lambda row:
                                                    calc_angle_3_gps_points(row[['prev_lat', 'prev_lon']].values,
                                                                            row[['latitude', 'longitude']].values,
                                                                            row[['next_lat', 'next_lon']].values)
                                                    if (not math.isnan(row['prev_lat'])) &
                                                       (not math.isnan(row['next_lat']))
                                                    else None,
                                                    axis=1)
    return trip_record


def moving_angle_bin_statistic(trip_record):
    """
    Create the bins and calc the statistic
    :param trip_record: data frame with 'moving_angle' col
    :return:
    """
    trip_record = calc_moving_angle_for_trip(trip_record.copy())
    trip_record = create_global_features_group_1_attr(trip_record)
    bins = (0, 91, 135, 180)
    cols = ['speed', 'speed_diff', 'acceleration_est_1', 'acceleration_diff']
    agg_function = ['mean', 'std', 'min', 'max']
    des = trip_record.groupby(pd.cut(trip_record["moving_angle"], bins, right=True))[cols].agg(agg_function).fillna(0)\
        .stack().reset_index().drop(['level_1', 'moving_angle'], axis=1)
    array = des.values.reshape((-1, des.shape[0]*des.shape[1]))
    ind_names = []
    for b in bins[1:]:
        for agg in agg_function:
            for col in cols:
                ind_names.append(str(b)+'_'+agg+'_'+col)
    df = pd.DataFrame(data=array, columns=ind_names)
    return df


# def percentile(n):
#     def percentile_(x):
#         return np.percentile(x, n)
#     percentile_.__name__ = 'percentile_%s' % n
#     return percentile_

def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:02.0f}'.format(n*100)
    return percentile_


def road_type_statistics(trip_record):
    trip_record_no_na = trip_record.copy().loc[~(trip_record.road_speed.isna())]

    cols = ['speed', 'speed_diff', 'acceleration_est_1', 'acceleration_diff', 'angular_acc']
    agg_function = ['mean', 'std', 'min', percentile(0.25), percentile(0.50), percentile(0.75), 'max']
    agg_function_str = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    bins = (0, 50, 70, 90, 130)
    ind_names = []
    for b in bins[1:]:
        for agg in agg_function_str:
            for col in cols:
                ind_names.append(str(b)+'_'+agg+'_'+col)

    if trip_record_no_na.shape[0] == 0:
        print('no road speed for trip')
        data = [0]*(len(bins)-1)*len(agg_function_str)*len(cols)
        print(len(data))
        print(len(ind_names))
        return pd.DataFrame(data=data, index=ind_names).T

    else:
        trip_records_attr = create_global_features_group_1_attr(trip_record_no_na)
        #print(trip_records_attr)
        des = trip_records_attr.groupby(pd.cut(trip_records_attr.road_speed, bins, right=True))[cols]\
            .agg(agg_function).fillna(0).stack().reset_index().drop(['level_1', 'road_speed'], axis=1)
        array = des.values.reshape((-1, des.shape[0] * des.shape[1]))
        df = pd.DataFrame(data=array, columns=ind_names)
        return df


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    :param lon1: longitude GPS of early point
    :param lat1: latitude GPS of early point
    :param lon2: longitude GPS of late point
    :param lat2: latitude GPS of late point
    :return: the distance of the points on circle.
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km


def minimal_rectangle_trip(trip_records):
    """
    Find the minimal rectangle containing the trip ad calc its area and edges lengths.
    :param trip_records: series of gps data of a trip
    :return: edges length and area values of the trip
    """
    min_lat, max_lat = trip_records['latitude'].agg(['min', 'max'])
    min_long, max_long = trip_records['longitude'].agg(['min', 'max'])
    h_edge = haversine(min_long, min_lat, min_long, max_lat)  # A,D
    v_edge = haversine(min_long, min_lat, max_long, min_lat)  # A,B
    return h_edge, v_edge, h_edge*v_edge, (min_lat, max_lat, min_long, max_long)


def get_day_date_string(num_str, i):
    if int(num_str)+i*7 < 10:
        return "0"+str(int(num_str)+i*7)
    else:
        return str(int(num_str)+i*7)


def read_same_trips(start_data_day, start_data_month, start_data_year, weeks_num=3,
                    directory='C:\\Users\\Shachaf\\Documents\\Shachaf\\sixt\\data_temp\\'):
    """
    Read data files of the same day of 3 subsequent weeks.
    :return: Drives data frame, GPS data frame and active vehicle on all the days ids.
    """
#     print('read_same_trips of %d weeks' % weeks_num)
    drives_lst, gps_lst = [], []
    d_count = 0
    g_count = 0
    for i in range(weeks_num):
#         print('Reads week %d' % i)
        drives_lst.append(pd.read_csv(directory + 'DRIVES/%s/%s/%s/SIXT_%s_%s_%s_DRIVES.csv.gz'
                                      % (start_data_year, start_data_month, get_day_date_string(start_data_day, i),
                                         start_data_year, start_data_month, get_day_date_string(start_data_day, i)),
                                      compression='gzip', header=0))
        d_count = d_count + drives_lst[i].shape[0]
        
        try:
            gps_lst.append(pd.read_csv(directory + 'GPS/%s/%s/%s/SIXT_%s_%s_%s_GPS.csv.gz'
                                   % (start_data_year, start_data_month, get_day_date_string(start_data_day, i),
                                      start_data_year, start_data_month, get_day_date_string(start_data_day, i)),
                                   compression='gzip', header=0))
        ###########
        except FileNotFoundError as not_found:
#             print('exception: more than one sub-folder')
            father_dir = os.walk(directory + 'GPS/%s/%s/%s/' % (start_data_year, start_data_month, get_day_date_string(start_data_day, i)))
            sub_dirs = [x[0] for x in father_dir]
            sub_dirs = sub_dirs[1:]
            for sub_dir in sub_dirs:
#                 print(sub_dir,sub_dir[-2:])
                gps_lst.append(pd.read_csv(sub_dir+'/SIXT_%s_%s_%s_%s_GPS.csv.gz'
                                              % (start_data_year, start_data_month, get_day_date_string(start_data_day, i),sub_dir[-2:]),
                                              compression='gzip', header=0))
            
        
        if len(gps_lst) > 0:
            g_count = g_count + gps_lst[-1].shape[0]
            #g_count = g_count + gps_lst[i].shape[0]
        
        
        if i == 0:
            active_vehicles = list(set(drives_lst[0].iloc[:,1].unique()))#['VEHICLE_ID'].unique()))
        else:
            active_vehicles = list(set(active_vehicles) & set(drives_lst[i].iloc[:,1].unique()))#['VEHICLE_ID'].unique()))

        new_columns=[]
        for col in drives_lst[-1].columns:
            if isinstance(col, str):
                col=col.upper()
            new_columns.append(col)
        drives_lst[-1].columns = new_columns

            
        #drives_lst.columns = [column.upper() for column in drives_lst.columns]
    drives_df = pd.concat(drives_lst)

    gps_df = pd.concat(gps_lst,axis=0)

#     print('Drives df vs cum: ', drives_df.shape[0], ', ', d_count)
#     print('GPS df vs cum: ', gps_df.shape[0], ', ', g_count)
    return drives_df, gps_df, active_vehicles


def read_all_files_in_folder(path):
    """
    The function reads all the files in a given folder and reduce duplicate rows
    (for creation of the vehicle-model table).
    :param path: string of the folder path where the files are.
    :return: a data frame with the files data with no duplicated rows.
    """
    dfs_lst = []
    count_rows = 0
    for filename in os.listdir(path):
        if filename.endswith(".csv.gz"):
            df = pd.read_csv(path + '\\' + filename, compression='gzip', header=0)
            dfs_lst.append(df)
            count_rows += df.shape[0]
        else:
            print('No .csv.gz file suffix')
    concat_df = pd.concat(dfs_lst)
    print('Are the rows match? - ', concat_df.shape[0] == count_rows, '\n Rows count = ', count_rows)
    concat_df_no_dupe = concat_df.drop_duplicates().reset_index(drop=True)
    return concat_df_no_dupe


def create_vehicle_model_table(input_path, output_path):
    """
    Uses the read_all_files_in_folder to create the table of vehicle_models as csv file.
    :param input_path: string of the folder path where the VCLS files are
    :param output_path: string of the folder path where the df be saved
    :return: the dataframe
    """
    df = read_all_files_in_folder(input_path).drop_duplicates(['VEHICLE_ID']).dropna(how='any')
    df.index = df['VEHICLE_ID']
    df.to_csv(output_path + '\\Vehicles_Models.csv.gz', compression='gzip')
    return df


# Read the model-vehicle table from the memory.
# df_up = pd.read_csv('C:\\Users\\Shachaf\\Documents\\Shachaf\\sixt\\Aggregation Data\\Vehicles_Models.csv.gz',
#                    compression='gzip', header=0, index_col ='VEHICLE_ID.1')


def separate_to_drives(drives_df, gps_df):
    gps_df = gps_df.sort_values(['orig_time'])
    gps_df = gps_df.reset_index().drop(['index'], axis=1)
    try:
        drives_df = drives_df.sort_values(['VEHICLE_ID', 'START_DRIVE'])
    except:
        drives_df = drives_df.sort_values(['vehicle_id', 'start_drive'])
    vehicle_trips_lst = []
    many_vehicles_trips_lst = []
    vehicle_num = 0

    for index, row in drives_df.iterrows():
        try:
            drive_id = row['DRIVE_ID']
            v_id = row['VEHICLE_ID']
        except:
            drive_id = row['drive_id']
            v_id = row['vehicle_id']
            
        if v_id != vehicle_num:
            many_vehicles_trips_lst.append(vehicle_trips_lst)
            print('Finished Trips of vehicle. Number of trips: %d' % len(vehicle_trips_lst))
            vehicle_num = v_id
            print('Start Trips of vehicle %d' % vehicle_num)
            vehicle_trips_lst = []
        try:
            trip_df = gps_df.loc[(gps_df['orig_time'] >= row['START_DRIVE']) &
                                 (gps_df['orig_time'] <= row['END_DRIVE']) &
                                 (gps_df['vehicle_id'] == row['VEHICLE_ID'])]
        except:
            trip_df = gps_df.loc[(gps_df['orig_time'] >= row['start_drive']) &
                                 (gps_df['orig_time'] <= row['end_drive']) &
                                 (gps_df['vehicle_id'] == row['vehicle_id'])]
        trip_df = trip_df.drop_duplicates()
        # trip_df = trip_df.loc[trip_df['vehicle_state'] == 1]
        if trip_df.empty:
            continue
        else:
            #dir_name = 'C:\\Users\\Shachaf\\Documents\\Shachaf\\sixt\\Aggregation Data\\TripGBDT\\Trips Data\\%d' \
            #          % vehicle_num
            #if not os.path.exists(dir_name):
            #    os.makedirs(dir_name)
            #trip_df.to_csv(dir_name + '\\%d.csv.gz' % drive_id, compression='gzip', header=0)
            trip_inst = Trip(drive_id, trip_df)
            vehicle_trips_lst.append(trip_inst)
    many_vehicles_trips_lst.append(vehicle_trips_lst)
    print('Finished Trips of vehicle. Number of trips: %d' % len(vehicle_trips_lst))
    return many_vehicles_trips_lst[1:]


def speed_estimation(trip_record):
    """
    Estimate the speed with the GPS points and calculate some statistical measures of the estimation.
    :param trip_record: data frame with the original GPS meta-data
    :return: trip_record data frame with new columns - prev_time, prev_latitude, prev_longitude
    """
    trip_record['prev_time'] = trip_record['orig_time'].shift(1)
    trip_record['prev_latitude'] = trip_record['latitude'].shift(1)
    trip_record['prev_longitude'] = trip_record['longitude'].shift(1)
    trip_record['speed_est'] = trip_record.apply(lambda row: haversine(row['prev_longitude'], row['prev_latitude'],
                                                                       row['longitude'], row['latitude']) /
                                                             ((row['orig_time']-row['prev_time']).seconds/(60*60)),
                                                 axis=1)
    pearson_corr, p_value = pearsonr(trip_record['speed_est'].iloc[1:], trip_record['speed'].iloc[1:])
    plt.plot(trip_record['orig_time'], trip_record['speed_est'])
    plt.plot(trip_record['orig_time'], trip_record['speed'])
    plt.show()
    return trip_record, (pearson_corr, p_value)


def acceleration_estimation(trip_record):
    """
    Estimate the acceleration with the speed data.
    :param trip_record: data frame with the original GPS meta-data
    :return: trip_record data frame with new columns - prev_time, next_time, prev_speed, next_speed,
                acceleration_est_1/2/3
    """
    trip_record['prev_time'] = trip_record['orig_time'].shift(1)
    trip_record['next_time'] = trip_record['orig_time'].shift(-1)
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
    trip_record['angular_acc'] = trip_record.apply(lambda row:
                                                   (row['next_direction'] - 2 * row['direction'] + row['prev_direction']) /
                                                   ((row['next_time'] - row['orig_time']).seconds * (row['orig_time'] - row['prev_time']).seconds), axis=1)
    return trip_record


def get_statistic(data_frame, column_name_str):
    """
    Create the statistical measures of the column - mean, std, min, max, and 25%, 50% and 75% quantiles.
    :param data_frame: data frame with the desired col.
    :param column_name_str: col for statistic
    :return: data frame with 1 row with the statistical measures (with index)
    """
    return pd.DataFrame(data_frame[column_name_str].describe().drop('count').values,
                        index=[x+'_'+column_name_str for x in data_frame[column_name_str].describe()
                        .drop('count').index]).T


def create_global_features_group_1(trip_records_no_acc):
    """
    Creation of Group 1 Global Features TODO: add angular_acceleration_features - change in degree
    :param trip_records_no_acc: data frame separation of trip with original meta-data
    :return: row of data frame with the features.
    """
    trip_acc = create_global_features_group_1_attr(trip_records_no_acc)
    # trip_acc = acceleration_estimation(trip_records_no_acc)
    speed_norm_features = get_statistic(trip_acc, 'speed')
    # trip_acc['speed_diff'] = trip_acc['speed'] - trip_acc['prev_speed']
    speed_diff_features = get_statistic(trip_acc, 'speed_diff')
    acceleration_norm_features = get_statistic(trip_acc, 'acceleration_est_1')
    # trip_acc['acceleration_diff'] = trip_acc['acceleration_est_1'] - trip_acc['acceleration_est_1'].shift(1)
    acceleration_diff_features = get_statistic(trip_acc, 'acceleration_diff')
    angular_acceleration_features = get_statistic(trip_acc, 'angular_acc')
    df_row = pd.concat(
        [speed_norm_features, speed_diff_features, acceleration_norm_features, acceleration_diff_features,
         angular_acceleration_features], axis=1)
    return df_row


def create_global_features_group_1_attr(trip_records_no_acc):
    """
    Create the necessary attributes for the group.
    :param trip_records_no_acc: data frame with the GPS meta data
    :return: data frame with additional 3 column - acceleration_est_1, speed_diff, acceleration_diff, angular_acc
    """
    trip_acc = acceleration_estimation(trip_records_no_acc)
    trip_acc_ang = angular_acc_estimation(trip_acc)
    trip_acc_ang['speed_diff'] = trip_acc_ang['speed'] - trip_acc_ang['prev_speed']
    trip_acc_ang['acceleration_diff'] = trip_acc_ang['acceleration_est_1'] - trip_acc_ang['acceleration_est_1'].shift(1)
    return trip_acc_ang


def create_global_features_group_2(trip_records):
    """
    Creation of Group 2 Global Features
    :param trip_records: data frame separation.
    :return: 6 features - trip time duration (1), trip length (2), average speed (2/1), area of the minimal rectangle
             containing the trip shape, and lengths of the two edges of the minimal rectangle.
    """
    trip_duration_hour = (trip_records['orig_time'].max() - trip_records['orig_time'].min()).seconds/(60*60)
    trip_length_km = trip_records['mileage'].max() - trip_records['mileage'].min()
    trip_average_speed = trip_length_km/trip_duration_hour
    length_edge_h, length_edge_v, area, coordinates = minimal_rectangle_trip(trip_records)
    df_row = pd.DataFrame((trip_duration_hour, trip_length_km, trip_average_speed, area, length_edge_h, length_edge_v),
                          index=['trip_duration_h', 'trip_length_km', 'trip_average_speed',
                                 'area_rec', 'length_edge_h', 'length_edge_v'])
    return df_row.T


def create_idle_feature(trip_record):
    """
    Compute the percentage of time the vehicle was idle
    :param trip_record: with both the 1 and 2 vehicle_state
    :return: data frame with one row of feature idle_ratio
    """
    idle_ratio = trip_record.loc[trip_record['vehicle_state'] == 2].count()[0] / trip_record.shape[0]
    d = {'idle_ratio': [idle_ratio]}
    idle = pd.DataFrame(data=d)
    return idle


def create_road_type_ratio(trip_record):
    """
    Identify the trip type - only in city trip, long trip out of town, mixed of both.
    :param trip_record: data frame with no idle sample.
    :return: 5 features that sums the ratio of time being driving on road type
    """
    trip_record_no_na = trip_record.copy().loc[~(trip_record.road_speed.isna())]
    if trip_record_no_na.shape[0] == 0:
        print('no road speed for trip')
        return pd.DataFrame(data=[0, 0, 0, 0], index=['(0, 50]', '(50, 70]', '(70, 90]', '(90, 130]']).T
    else:
        bins = (0, 50, 70, 90, 130)
        road_type_ratio = \
            (trip_record_no_na.groupby(pd.cut(trip_record_no_na.road_speed, bins, right=True)).count()['vehicle_id']) /\
            trip_record_no_na.shape[0]
        df = pd.DataFrame(road_type_ratio).T
        df.columns = ['(0, 50]', '(50, 70]', '(70, 90]', '(90, 130]']
        return df.reset_index().drop(['index'], axis=1)


def create_events_features(drive_id, drives_data):
    try:
        events_lst = ['TURN1', 'TURN2', 'TURN3', 'BREAK1', 'BREAK2', 'BREAK3', 'ACCELERATION1', 'ACCELERATION2',
                  'ACCELERATION3', 'SPEED1', 'SPEED2', 'SPEED3']
        event_features = drives_data.loc[drives_data['DRIVE_ID'] == drive_id][events_lst]
        event_features['sum_events'] = event_features.sum(axis=1)
    except:
        events_lst = ['turn1', 'turn2', 'turn3', 'break1', 'break2', 'break3', 'acceleration1', 'acceleration2',
                  'acceleration3', 'speed1', 'speed2', 'speed3']
        event_features = drives_data.loc[drives_data['drive_id'] == drive_id][events_lst]
        event_features['sum_events'] = event_features.sum(axis=1)
    return event_features.reset_index()


def create_features(trips, driver_num, drives_data):
    print('create global features')
    records = []
    trip_ids = []
    in_if = 0
    for i in tqdm(range(driver_num)):
        j = 0
        for trip_j in trips[i]:
            trip = trip_j.time_series_record
            trip_id = trip_j.id
            # print('i:', i)
            # print('j:', j)
            trip_no_idle = trip.loc[trip['vehicle_state'] == 1].copy()
            if trip_no_idle.empty or trip_no_idle.shape[0] < 10:
                in_if = in_if+1
                continue
            # print('i is ', i, 'j is ', j)
            f_1 = create_global_features_group_1(trip_no_idle)
            f_2 = create_global_features_group_2(trip_no_idle)
            f_3 = create_idle_feature(trip)
            f_4 = moving_angle_bin_statistic(trip_no_idle)
            f_5 = create_road_type_ratio(trip_no_idle)
            f_6 = road_type_statistics(trip_no_idle)
            f_7 = create_events_features(trip_id, drives_data)
            try:
                f_8 = drives_data.loc[drives_data['DRIVE_ID'] == trip_id][['DRIVE_ID', 'START_DRIVE', 'END_DRIVE', 'START_LATITUDE', 'START_LONGITUDE', 'END_LATITUDE', 'END_LONGITUDE']].reset_index(drop=True)
            except:
                f_8 = drives_data.loc[drives_data['drive_id'] == trip_id][['drive_id', 'start_drive', 'end_drive', 'start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']].reset_index(drop=True)
            row = pd.concat([f_8, f_1, f_2, f_3, f_4, f_5, f_6, f_7], axis=1, sort=False)
            row['vehicle_id'] = trip['vehicle_id'].iloc[0]
            if j == 0:
                shape = row.shape
            if shape != row.shape:
#                 print('i is ', i, 'j is ', j)
                 print('first row ', shape)
#                 print('another row', row.shape)
            records.append(row)
            trip_ids.append(trip_id)
            j = j+1
        #if i % 100 == 0:
        #    with open('Aggregation Data\\TripGBDT\\trips_records_'+str(i)+'_driver_april_part.pickle', 'wb') as handle_:
        #        pickle.dump(records, handle_)
#     print(len(records))
#     print('count in if: ', in_if)
    df = pd.concat(records, axis=0)
    # df.index = trip_ids
    # df = df.drop(['index'], axis=1)
    return df
from tqdm import tqdm

def Prepare_metadata_by_vehicle_number(vehicle_ids):
    PATH_raw_data = '/data/inputs/Year_Data'
    vehicle_ids = [vehicle_ids]
    
    for year_str in ['2018','2019']:
        for month_index in tqdm(range(1, 13, 1)):
            if os.path.exists(f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{month_index}_{year_str}_{vehicle_ids}_trips.pickle")\
    and os.path.exists(f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{month_index}_{year_str}_{vehicle_ids}_car.csv"):
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
                    f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{month_index}_{year_str}_{vehicle_ids}_trips.pickle",
                    'wb') as handle:
                pickle.dump(trips_lst, handle)
            df_global = create_features(trips_lst, len(trips_lst), drives_sub_group)
            df_global = df_global.fillna(0)
            df_global = df_global.replace([np.inf, -np.inf], np.nan).dropna(how='any')

            df_global.to_csv(
                f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{month_index}_{year_str}_{vehicle_ids}_car.csv",
                mode='w', index=False)

def Prepare_coors_metadata_by_coors_lst( directory_d='/data/inputs/Year_Data/DRIVES', directory_g='/data/inputs/Year_Data/GPS', coor_lst=[]):
    
    """
    Export data based on coordinates and date.

    Args:
        \
        directory_d (str): The directory for drive data (default is '/data/inputs/Year_Data/DRIVES').
        directory_g (str): The directory for GPS data (default is '/data/inputs/Year_Data/GPS').
        coor_lst (list): A list of coordinates to filter the data (list of cors presented like tuples).

    Returns:
        None
    """
    
    for year in ['2018','2019']:
        for month in range(1,13,1): 
            if len(str(month))==1:
                # Create the path for the year and month
                year_path_d = os.path.join(directory_d, str(year))
                month_path_d = os.path.join(year_path_d, '0'+str(month))
                year_path_g = os.path.join(directory_g, str(year))
                month_path_g = os.path.join(year_path_g, '0'+str(month))
            else:
                year_path_d = os.path.join(directory_d, str(year))
                month_path_d = os.path.join(year_path_d, str(month))
                year_path_g = os.path.join(directory_g, str(year))
                month_path_g = os.path.join(year_path_g, str(month))




    #         # Check if the year and month directories exist
    #         if not os.path.exists(year_path_d) or not os.path.exists(month_path_d):
    #             print("Year or month not found.")
    #             return

            # List all the day folders
            days = os.listdir(month_path_d)

            # List to store the DataFrames
            dfs_d = []
            dfs_g = []
            coor_fixed = []
            for cor in coor_lst:
                if not os.path.exists(f"/bigdata/users-home/dor/transpotation research/agg_data/coors_data/{cor}/{month}_{year}.csv"):
                    coor_fixed.append(cor)
                
            if len(coor_fixed)>0:
                for day in tqdm(days[:20], f"days in {month}/{year}"):
                    try:
                        day_path_d = os.path.join(month_path_d, day)
                        day_path_g = os.path.join(month_path_g, day)

                        # List all the CSV files in the day folder
                        try:
                            csv_files_d = [f for f in os.listdir(day_path_d) if f.endswith('.csv.gz')][0]
                            csv_files_d = os.path.join(day_path_d, csv_files_d)
                            df_drives = pd.read_csv(csv_files_d)
                        except:
                            lst = []
                            csv_files_d = [f for f in os.listdir(day_path_d)]

                            # Iterate over CSV files
                            for path in csv_files_d:
                                csv_dd = os.path.join(day_path_d, path)
                                csv_d = [f for f in os.listdir(csv_dd) if f.endswith('.csv.gz')][0]
                                csv_dd = os.path.join(csv_dd, csv_d)
                                lst.append(pd.read_csv(csv_dd))
                            df_drives = pd.concat(lst, ignore_index=True)

                        try:
                            csv_files_g = [f for f in os.listdir(day_path_g) if f.endswith('.csv.gz')][0]
                            csv_files_g = os.path.join(day_path_g, csv_files_g)
                            df_gps = pd.read_csv(csv_files_g)
                        except:
                            lst = []
                            csv_files_g = [f for f in os.listdir(day_path_g)]

                            # Iterate over CSV files
                            for path in csv_files_g:
                                csv_gg = os.path.join(day_path_g, path)
                                csv_g = [f for f in os.listdir(csv_gg) if f.endswith('.csv.gz')][0]
                                csv_gg = os.path.join(csv_gg, csv_g)
                                lst.append(pd.read_csv(csv_gg))
                            df_gps = pd.concat(lst, ignore_index=True)
                    except:
                        continue


                    df_drives.columns = df_drives.columns.str.lower()
                    df_gps.columns = df_gps.columns.str.lower()


                    df_drives["corrs"] = df_drives.apply(lambda row: (round(row["start_latitude"], 2),
                                                                     round(row["start_longitude"], 2),
                                                                     round(row["end_latitude"], 2),
                                                                     round(row["end_longitude"], 2)), axis=1)


                    df_drives = df_drives[df_drives["corrs"].isin(coor_fixed)]


                    # Convert datetime columns to datetime objects
                    df_drives['start_drive'] = pd.to_datetime(df_drives['start_drive'])
                    df_drives['end_drive'] = pd.to_datetime(df_drives['end_drive'])
                    df_gps['orig_time'] = pd.to_datetime(df_gps['orig_time'])

                    # Append DataFrames to the list
                    dfs_d.append(df_drives)
                    dfs_g.append(df_gps)

                # Check if any DataFrames were found
                if len(dfs_d) == 0:

                    print("No CSV files found.")
                else:
                    # Concatenate the DataFrames
                    concatenated_df_d = pd.concat(dfs_d, ignore_index=True)
                    concatenated_df_g = pd.concat(dfs_g, ignore_index=True)
                    
                    # Transform all columns to upper letters
                    if len(concatenated_df_d) >= 1:
                        # Assuming you have a function called 'separate_to_drives' defined elsewhere
                        drives_sub_group = concatenated_df_d[concatenated_df_d.drive_duration >= 10]
                        trips_lst = separate_to_drives(drives_sub_group, concatenated_df_g)

                        df_global = create_features(trips_lst, len(trips_lst), drives_sub_group)
                        df_global["corrs"] = df_global.apply(lambda row: (round(row["start_latitude"], 2),
                                                                     round(row["start_longitude"], 2),
                                                                     round(row["end_latitude"], 2),
                                                                     round(row["end_longitude"], 2)), axis=1)
                        df = df_global.copy()
                        for cor in coor_fixed:
                            df = df_global[df_global['corrs'] == cor]
                            try:
                                df = df.fillna(0)
                                df = df.replace([np.inf, -np.inf], np.nan).dropna(how='any')
                                # Define the directory path
                                directory_path = f"/bigdata/users-home/dor/transpotation research/agg_data/coors_data/{cor}"

                                # Create the directory if it doesn't exist
                                if not os.path.exists(directory_path):
                                    print("making dir for",cor )
                                    os.makedirs(directory_path)
                                df.to_csv(f"/bigdata/users-home/dor/transpotation research/agg_data/coors_data/{cor}/{month}_{year}.csv", mode='w', index=False)

                            except:
                                print('moving to next month')
                            
                            
                            
def create_features_for_coor(coors, coors_dir_path="/bigdata/users-home/dor/transpotation research/agg_data/coors_data"):
    """
    Create features for a specified set of coordinates.

    This function reads CSV files containing data for the given coordinates, processes the data, and computes
    statistical features. It stores the computed features in a pickle file for future use.

    Args:
        coors (tuple): A tuple of coordinates (START_LATITUDE, START_LONGITUDE, END_LATITUDE, END_LONGITUDE).
        coors_dir_path (str): The directory containing CSV data files for the specified coordinates.

    Returns:
        list: A list containing three DataFrames:
            1. df_count: DataFrame with count statistics for coordinate occurrences.
            2. df_mean: DataFrame with mean statistics for coordinate data.
            3. df_std: DataFrame with standard deviation statistics for coordinate data.
    """
    # Check if features for the coordinates already exist in a pickle file
    if os.path.exists(f"/bigdata/users-home/dor/transpotation research/agg_data/features_coors/{coors}.pickle"):
        with open(f"/bigdata/users-home/dor/transpotation research/agg_data/features_coors/{coors}.pickle", 'rb') as handle:
            loaded_data = pickle.load(handle)
        return loaded_data
    
        
        # Check if CSV files exist for the specified coordinates
    elif os.path.exists(os.path.join(coors_dir_path, f'{coors}')):
            
        csv_files = [f for f in os.listdir(os.path.join(coors_dir_path, f'{coors}')) if f.endswith('.csv')]
    else:
        Prepare_coors_metadata_by_coors_lst(coor_lst=coors)

        csv_files = [f for f in os.listdir(os.path.join(coors_dir_path, f'{coors}')) if f.endswith('.csv')]

    # Read and concatenate CSV files for the coordinates
    df_lst = []
    for month in csv_files:
        file = os.path.join(os.path.join(coors_dir_path, f'{coors}'), month)
        try:
            df_lst.append(pd.read_csv(file))
        except:
            continue
    df = pd.concat(df_lst, ignore_index=True)

    # Round coordinate values to 2 decimal places for consistency
    df["start_latitude"] = df["start_latitude"].round(2)
    df["start_longitude"] = df["start_longitude"].round(2)
    df["end_latitude"] = df["end_latitude"].round(2)
    df["end_longitude"] = df["end_longitude"].round(2)

    # Filter the DataFrame for the specified coordinates
    df = df[(df['start_latitude'] == coors[0]) &
            (df['start_longitude'] == coors[1]) &
            (df['end_latitude'] == coors[2]) &
            (df['end_longitude'] == coors[3])]
#     print(df[['corrs', 'speed3',  'speed1', 'break3', 'acceleration1', 'speed2', 'acceleration2', 'index', 'turn2', 'break1', 'sum_events', 'drive_id', 'acceleration3', 'turn3', 'turn1', 'break2']].head())
#     df_mean1 = df.groupby(['vehicle_id','start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']).mean().reset_index()

    if len(df) > 10:
        # Compute count, mean, and standard deviation statistics for the coordinates
        df_count = df.groupby(['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']).count().reset_index()
        df_mean = df.groupby(['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']).mean().reset_index()
        df_std = df.groupby(['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']).std().reset_index()

        # Save the computed features in a pickle file
        with open(f"/bigdata/users-home/dor/transpotation research/agg_data/features_coors/{coors}.pickle", 'wb') as handle:
            pickle.dump([df_count, df_mean, df_std], handle)

        # Return the computed features
        return [df_count, df_mean, df_std]
    else:
        with open(f"/bigdata/users-home/dor/transpotation research/agg_data/features_coors/{coors}.pickle", 'wb') as handle:
            pickle.dump(['less then 10 rows'], handle)
        return ['less then 10 drives here']
def normalize_columns(row, feat_dict, df):
    coordinates = (row['start_latitude'], row['start_longitude'], row['end_latitude'], row['end_longitude'])
    if coordinates in feat_dict:
        df_mean = feat_dict[coordinates][1]
        df_std = feat_dict[coordinates][2]
        
        if feat_dict[coordinates][0].iloc[0, 5] < 5:
            # Delete the row from the original DataFrame df
            df.drop(row.name, inplace=True)
        else:
            for col in df.columns:
                if col in df_mean.columns and col not in ['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude', 'vehicle_id', 'drive_id']:
                    mean = df_mean.at[0, col]
                    std = df_std.at[0, col]
                    if std != 0:
                        row[col] = (row[col] - mean) / std
    return row


# In[53]:


def scaling_by_coors_feat(df,car_id=460631):
    cors_lst = list(df.apply(lambda row: (round(row["start_latitude"], 2), round(row["start_longitude"], 2), round(row["end_latitude"], 2), round(row["end_longitude"], 2)), axis=1))
    cors_lst = list(set(cors_lst))
    df["corrs"] = df.apply(lambda row: (round(row["start_latitude"], 2),
                                                                     round(row["start_longitude"], 2),
                                                                     round(row["end_latitude"], 2),
                                                                     round(row["end_longitude"], 2)), axis=1)
    cols_to_exclude = ['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude', 'vehicle_id', 'drive_id','end_drive','start_drive','corrs','index','start_hour','end_hour','date','day_of_week','turn1','turn2','corrs', 'speed3',  'speed1', 'break3', 'acceleration1', 'speed2', 'acceleration2', 'index', 'turn2', 'break1', 'sum_events', 'drive_id', 'acceleration3', 'turn3', 'turn1', 'break2'] 
    feat_dict = {}
    for cor in cors_lst:
        feat_dict[tuple(cor)] = create_features_for_coor(tuple(cor))
        #print(feat_dict[tuple(cor)])
    for i in range(len(df)):
        for cor in feat_dict:
            if df['corrs'].iloc[i]==cor:
                for column in df.columns:
                    if column not in cols_to_exclude and type(feat_dict[cor][0])!=str:
                        if feat_dict[cor][2].at[0,column]==0:
                            divider = 1
                        else: 
                            col_index =  [i for i,y in enumerate(list(df.columns)) if y == column]
        #                     print(col_index)
                            divider = feat_dict[cor][2].at[0,column]
        #                     print(divider)
                        df.iloc[i,col_index[0]] = (df.iloc[i,col_index[0]] - feat_dict[cor][1][column].iloc[0])/divider
    #df = df.apply(normalize_columns, axis=1, args=(feat_dict,df))
    # Assuming df is your DataFrame and 'cols_to_exclude' contains columns to keep
     # Replace with your column names

#     mask = df.drop(cols_to_exclude, axis=1).apply(lambda x: (x > 4.5) | (x < -4.5)).any(axis=1)
#     cleaned_df = df[~mask]

    df.to_csv(f"/bigdata/users-home/dor/transpotation research/agg_data/scaled_df/{int(car_id)}.csv", mode='w', index=False)

    return df


def export_processed_data_of_veich(car_id,month,year):
    path = f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{[car_id]}/{month}_{year}_car.csv"
    
    return pd.read_csv(path)




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
    return  pd.concat(df_lst)
                    
if __name__ == '__main__':
    drives1, gps1, vehicle_ids1 = read_same_trips('01', '04', '2018', 5)
    drives1['START_DRIVE'] = pd.to_datetime(drives1['START_DRIVE'])
    drives1['END_DRIVE'] = pd.to_datetime(drives1['END_DRIVE'])
    gps1['orig_time'] = pd.to_datetime(gps1['orig_time'])

    drives2, gps2, vehicle_ids2 = read_same_trips('02', '04', '2018', 5)
    drives2['START_DRIVE'] = pd.to_datetime(drives2['START_DRIVE'])
    drives2['END_DRIVE'] = pd.to_datetime(drives2['END_DRIVE'])
    gps2['orig_time'] = pd.to_datetime(gps2['orig_time'])

    drives3, gps3, vehicle_ids3 = read_same_trips('03', '04', '2018', 4)
    drives3['START_DRIVE'] = pd.to_datetime(drives3['START_DRIVE'])
    drives3['END_DRIVE'] = pd.to_datetime(drives3['END_DRIVE'])
    gps3['orig_time'] = pd.to_datetime(gps3['orig_time'])

    drives4, gps4, vehicle_ids4 = read_same_trips('04', '04', '2018', 4)
    drives4['START_DRIVE'] = pd.to_datetime(drives4['START_DRIVE'])
    drives4['END_DRIVE'] = pd.to_datetime(drives4['END_DRIVE'])
    gps4['orig_time'] = pd.to_datetime(gps4['orig_time'])

    drives5, gps5, vehicle_ids5 = read_same_trips('05', '04', '2018', 4)
    drives5['START_DRIVE'] = pd.to_datetime(drives5['START_DRIVE'])
    drives5['END_DRIVE'] = pd.to_datetime(drives5['END_DRIVE'])
    gps5['orig_time'] = pd.to_datetime(gps5['orig_time'])

    drives6, gps6, vehicle_ids6 = read_same_trips('06', '04', '2018', 4)
    drives6['START_DRIVE'] = pd.to_datetime(drives6['START_DRIVE'])
    drives6['END_DRIVE'] = pd.to_datetime(drives6['END_DRIVE'])
    gps6['orig_time'] = pd.to_datetime(gps6['orig_time'])

    drives7, gps7, vehicle_ids7 = read_same_trips('07', '04', '2018', 4)
    drives7['START_DRIVE'] = pd.to_datetime(drives7['START_DRIVE'])
    drives7['END_DRIVE'] = pd.to_datetime(drives7['END_DRIVE'])
    gps7['orig_time'] = pd.to_datetime(gps7['orig_time'])

    drives = pd.concat([drives1, drives2, drives3, drives4, drives5, drives6, drives7], ignore_index=True)  # To big for memory???
    gps = pd.concat([gps1, gps2, gps3, gps4, gps5, gps6, gps7], ignore_index=True)  # To big for memory???
    vehicle_ids = list(
        set(vehicle_ids1) & set(vehicle_ids2) & set(vehicle_ids3) & set(vehicle_ids4) & set(vehicle_ids5)
        & set(vehicle_ids6) & set(vehicle_ids7))

    # Multiple vehicles
    df_global = pd.read_csv(
        'C:\\Users\\Shachaf\\Documents\\Shachaf\\sixt\\Aggregation Data\\TripGBDT'
        '\\df_global_1000_vehicle_20_days_ang_acc_ext_bins.csv', header=0, index_col=0)

    vehicle_ids_sub_group = list(df_global['vehicle_id'].unique())

    drives_sub_group = drives.loc[drives['VEHICLE_ID'].isin(vehicle_ids_sub_group)]

    # Length of trip to include:
    trip_len_min = 10
    drives_sub_group = drives_sub_group.loc[drives_sub_group['DRIVE_DURATION'] >= trip_len_min]

    gps_sub_group = gps.loc[gps['vehicle_id'].isin(vehicle_ids_sub_group)]

    trips_lst = separate_to_drives(drives_sub_group, gps_sub_group)
    with open('Aggregation Data\\TripGBDT\\trips_lst_april_2018.pickle', 'wb') as handle:
        pickle.dump(trips_lst, handle)
    df_global = create_features(trips_lst, len(trips_lst), drives_sub_group)
    # df_global.index = df_global['trip_id']
    df_global = df_global.fillna(0)
    df_global = df_global.replace([np.inf, -np.inf], np.nan).dropna(how='any')

    df_global.to_csv(
        'C:\\Users\\Shachaf\\Documents\\Shachaf\\sixt\\Aggregation Data\\TripGBDT\\1000_vcls_april_2018_whole_trip_features.csv')

'''
df_global = pd.read_csv(
    'C:\\Users\\Shachaf\\Documents\\Shachaf\\sixt\\Aggregation Data\\TripGBDT\\df_global_1000_vehicle_20_days_ang_acc_ext_bins.csv',
    header=0, index_col=0)

vcls_models_df = pd.read_csv(
    'C:\\Users\\Shachaf\\Documents\\Shachaf\\sixt\\Aggregation Data\\Vehicles_Models.csv.gz',
    header=0, index_col=1, compression='gzip')

mean_accuracy, test_trips_count, cm_all_v, thresh_accuracy, features_importance, all_vehicle_acc_dict,\
           model_str, n_vcls_ids_list = train_n_vehicle_models(50, df_global, vcls_models_df, model='GB')

features_names = df_global.drop(['vehicle_id'], axis=1).columns

mean_accuracy, cm_all_vehicles, features_importance_list, thresh_accuracy, all_vehicle_acc_dict, test_total_trips_count,\
           n_vcls_ids = results_outputs_all_vehicle(mean_accuracy, test_trips_count, cm_all_v, thresh_accuracy,
                                                    features_importance, all_vehicle_acc_dict, model_str,
                                                    n_vcls_ids_list, features_names)

'''
'''
t_1 = time.time()
# TODO work with meter and seconds instead of km and hour for acceleration?
# Read the data of one day only:
head_directory = 'C:\\Users\\Shachaf\\Documents\\Shachaf\\sixt\\data_temp\\'
# gps_day = pd.read_csv(head_directory + 'GPS\\04\\SIXT_2018_03_04_GPS.csv.gz', compression='gzip', header=0)
# gps_day = gps_day.loc[gps_day['vehicle_state'] == 1]
# gps_day = gps_day.reset_index()
# gps_day['orig_time'] = pd.to_datetime(gps_day['orig_time'])
# print(gps_day.columns)

# drive_day = pd.read_csv(head_directory + 'Drives\\04\\SIXT_2018_03_04_DRIVES.csv.gz', compression='gzip', header=0)
# drive_day['START_DRIVE'] = pd.to_datetime(drive_day['START_DRIVE'])
# drive_day['END_DRIVE'] = pd.to_datetime(drive_day['END_DRIVE'])
# print(drive_day.columns)

# gps = gps_day.copy()
# drives = drive_day.copy()


# Read 5 days of 3 weeks: 15 days
# drives, gps, vehicle_ids = read_same_trips('04', '03', '2018', 1)

drives1, gps1, vehicle_ids1 = read_same_trips('04', '03', '2018', 4)
drives2, gps2, vehicle_ids2 = read_same_trips('05', '03', '2018', 4)
drives3, gps3, vehicle_ids3 = read_same_trips('06', '03', '2018', 4)
drives4, gps4, vehicle_ids4 = read_same_trips('07', '03', '2018', 4)
drives5, gps5, vehicle_ids5 = read_same_trips('08', '03', '2018', 4)

drives = pd.concat([drives1, drives2, drives3, drives4, drives5], ignore_index=True)
gps = pd.concat([gps1, gps2, gps3, gps4, gps5], ignore_index=True)
vehicle_ids = list(set(vehicle_ids1) & set(vehicle_ids2) & set(vehicle_ids3) & set(vehicle_ids4) & set(vehicle_ids5))

drives['START_DRIVE'] = pd.to_datetime(drives['START_DRIVE'])
drives['END_DRIVE'] = pd.to_datetime(drives['END_DRIVE'])
gps['orig_time'] = pd.to_datetime(gps['orig_time'])

vehicle_num = 1000

# Multiple vehicles
vehicle_ids_sub_group = sample(vehicle_ids, vehicle_num)
# vehicle_ids_sub_group = [384528, 412526, 487147, 457477, 381564, 410610, 423810, 490211, 489425, 442038, 427144]
# 1 vehicle
# max_df = drives.groupby(['VEHICLE_ID']).count()['DRIVE_ID'].reset_index()
# vehicle_ids_sub_group = [max_df.loc[max_df['DRIVE_ID'] == max_df['DRIVE_ID'].max()]['VEHICLE_ID'].iloc[0]]


drives_sub_group = drives.loc[drives['VEHICLE_ID'].isin(vehicle_ids_sub_group)]
# Length of trip to include:
trip_len_min = 10
drives_sub_group = drives_sub_group.loc[drives_sub_group['DRIVE_DURATION'] >= trip_len_min]

gps_sub_group = gps.loc[gps['vehicle_id'].isin(vehicle_ids_sub_group)]

trips_lst = separate_to_drives(drives_sub_group, gps_sub_group)
# show_map(create_one_trip_map(trips_lst[1][0]), head_directory[:-11] + '\\Photos from code\\map.html')
df_global = create_features(trips_lst, len(trips_lst), drives_sub_group)
# df_global.index = df_global['trip_id']
df_global = df_global.fillna(0)
df_global = df_global.replace([np.inf, -np.inf], np.nan).dropna(how='any')

df_global.to_csv(
   'C:\\Users\\Shachaf\\Documents\\Shachaf\\sixt\\Aggregation Data\\TripGBDT\\df_global_1000_vehicle_20_days_ang_acc_ext_bins.csv')
'''

'''
# Threshold functions:   
df_global = pd.read_csv(
    'C:\\Users\\Shachaf\\Documents\\Shachaf\\sixt\\Aggregation Data\\TripGBDT\\df_global_1000_vehicle_20_days.csv',
    header=0, index_col=0)


df_global_no_inf = df_global.replace([np.inf, -np.inf], np.nan).dropna(how='any')
df_global_no_inf.head()
threshold_df = df_global_no_inf.copy()'''
'''

def road_type_th(row):
    if row['(30, 50]'] >= 0.68:
        return 0
    elif row['(80, 90]'] >= 0.33:
        return 1
    else:
        return None


threshold_df['road_type'] = threshold_df.apply(road_type_th, axis=1)
pd.value_counts(threshold_df['road_type'])
'''
# create_std_events_count_graph(threshold_df.loc[(threshold_df['road_type'] == 1) & (threshold_df['sum_events'] <= 20)],
#                               1000, 'sum_events', 'std_speed')

# events_cols_lst = ['sum_events', 'std_speed', 'mean_speed_diff', '90_mean_speed_diff', '90_std_speed',
#                   '135_mean_speed_diff', '135_std_speed', '180_mean_speed_diff', '180_std_speed']

# cor_event_std_speed = df_global.loc[df_global['sum_events'] > 0][events_cols_lst].corr()


# create_std_events_count_graph(df_global, vehicle_num, 'std_speed', 'sum_events')

# t_2 = time.time()
# print('Total seconds: ', round(t_2-t_1, 2))
# print('Total minutes: ', round((t_2-t_1)/60, 2))

'''
cols_names = list(df_global.columns)
kmeans_elbow_rule(df_global[cols_names[83:88]], 10, 'sse')
# plt.savefig('C:\\Users\\Shachaf\\Desktop\\Road type results\\elbow_rule.png')
plt.close()
kmeans_k = run_kmeans(df_global[cols_names[83:88]], 3)

kmeans_df_results = df_global.copy()
kmeans_df_results['cluster'] = kmeans_k.labels_

for col in cols_names[83:88]:
    kmeans_df_results.boxplot(col, by='cluster')
    plt.savefig('C:\\Users\\Shachaf\\Desktop\\Road type results\\' + col + '.png')
    plt.close()

kmeans_df_results.boxplot([cols_names[1]], by='cluster')
plt.savefig('C:\\Users\\Shachaf\\Desktop\\Road type results\\' + cols_names[1] + '.png')
plt.close()

plt.scatter(kmeans_df_results.loc[kmeans_df_results['cluster'] == 0]['(30, 50]'],
            kmeans_df_results.loc[kmeans_df_results['cluster'] == 0]['(80, 90]'])
plt.scatter(kmeans_df_results.loc[kmeans_df_results['cluster'] == 1]['(30, 50]'],
            kmeans_df_results.loc[kmeans_df_results['cluster'] == 1]['(80, 90]'])
plt.scatter(kmeans_df_results.loc[kmeans_df_results['cluster'] == 2]['(30, 50]'],
            kmeans_df_results.loc[kmeans_df_results['cluster'] == 2]['(80, 90]'])
plt.plot([0, 1], [1, 0], c='red')
plt.show()
'''
'''
model_df = data_prep_for_classifier(df_global)

t_2 = time.time()

print('Total seconds: ', round(t_2-t_1, 2))
print('Total minutes: ', round((t_2-t_1)/60, 2))

df = model_df.copy()
# x = df.drop(['label', 'vehicle_id', 'index'], axis=1)
# y = df['label']

knn_accuracy_no_scaled_validate, knn_accuracy_no_scaled_train, knn_accuracy_scaled_validate, knn_accuracy_scaled_train, \
knn_hyper_param_range = model_tuning(df, 'KNN', 10)
gbdt_accuracy_no_scaled_validate, gbdt_accuracy_no_scaled_train, gbdt_accuracy_scaled_validate, \
gbdt_accuracy_scaled_train, gbdt_hyper_param_range = model_tuning(df, 'GBDT', 150, 50)

plt.plot(knn_hyper_param_range, knn_accuracy_no_scaled_validate, '-', c='blue', label='no scale, v')
plt.plot(knn_hyper_param_range, knn_accuracy_no_scaled_train, '--', c='blue', label='no scale, t')
plt.plot(knn_hyper_param_range, knn_accuracy_scaled_validate, '-', c='orange', label='scale, v')
plt.plot(knn_hyper_param_range, knn_accuracy_scaled_train, '--', c='orange', label='scale, t')
plt.xlabel('k')
plt.xlabel('accuracy')
plt.legend()
plt.savefig('C:\\Users\\Shachaf\\Documents\\Shachaf\\sixt\\Photos from code\\TripGBDT\\Model Tuning\\knn_tuning_k')

plt.plot(gbdt_hyper_param_range, gbdt_accuracy_no_scaled_validate, '-', c='blue', label='no scale, v')
plt.plot(gbdt_hyper_param_range, gbdt_accuracy_no_scaled_train, '--', c='blue', label='no scale, t')
plt.plot(gbdt_hyper_param_range, gbdt_accuracy_scaled_validate, '-', c='orange', label='scale, v')
plt.plot(gbdt_hyper_param_range, gbdt_accuracy_scaled_train, '--', c='orange', label='scale, t')
plt.xlabel('n_estimators')
plt.xlabel('accuracy')
plt.legend()
plt.savefig(
    'C:\\Users\\Shachaf\\Documents\\Shachaf\\sixt\\Photos from code\\TripGBDT\\Model Tuning\\gbdt_tuning_n_estimators')
    '''
'''
clf, mean_score_v, mean_score_train = k_fold(df_global.drop(['vehicle_id'], axis=1).values, df_global['vehicle_id'].values)
print('score_mean_v', str(mean_score_v))
print('score_mean_train', str(mean_score_train))
important_features = df_global.drop(['vehicle_id'], axis=1).columns[clf.feature_importances_ >= 0.01]
print(important_features.shape)
cluster_df = df_global.copy()[important_features]
kmeans_elbow_rule(cluster_df, vehicle_num, 'sse')'''
"""
k = input('Type k:')
kmeans = run_kmeans(cluster_df, int(k))

# Project with TSNE
tsne_2 = TSNE(n_components=2, verbose=1, perplexity= 10, n_iter=1000)
tsne_results_2 = tsne_2.fit_transform(cluster_df)
df_tsne_2 = pd.DataFrame()
df_tsne_2['x-tsne'] = tsne_results_2[:, 0]
df_tsne_2['y-tsne'] = tsne_results_2[:, 1]
# Plot with TSNE projection
plt.scatter(df_tsne_2['x-tsne'], df_tsne_2['y-tsne'], c=kmeans.labels_)
plt.xlabel("x-tsne")
plt.ylabel("y-tsne")
plt.title("KMeans with T-SNE - Important Features")
plt.show()

cluster_df['label'] = kmeans.labels_
cluster_df['vehicle_id'] = df_global['vehicle_id']

# cluster_df.groupby(['vehicle_id']).nunique()['label']
# cluster_df.loc[]
"""
# TODO Read different day!
'''
# Function Tests
# One dFunction Tests
drive_sample = drive_day.iloc[1406]
print(drive_sample)
gps_sample = gps_day.loc[(gps_day['orig_time'] >= drive_sample['START_DRIVE']) &
                         (gps_day['orig_time'] <= drive_sample['END_DRIVE']) &
                         (gps_day['vehicle_id'] == drive_sample['VEHICLE_ID'])]
gps_sample = gps_sample.sort_values(['orig_time'])

# Function 1: minimal_rectangle_trip check
length_edge_h, length_edge_v, area, coordinates = minimal_rectangle_trip(gps_sample)

# Plot trip and minimal rectangle:
plt.scatter(gps_sample['latitude'], gps_sample['longitude'], s=10)
plt.plot([coordinates[0], coordinates[0]], [coordinates[2], coordinates[3]], c='red')
plt.plot([coordinates[1], coordinates[1]], [coordinates[2], coordinates[3]], c='red')
plt.plot([coordinates[0], coordinates[1]], [coordinates[2], coordinates[2]], c='red')
plt.plot([coordinates[0], coordinates[1]], [coordinates[3], coordinates[3]], c='red')
plt.show()
plt.close()

# Function 2: calc_angle_3_gps_points check
x = 1500
gps_3points = gps_day.iloc[[x, x+1, x+2]][['latitude', 'longitude']].values
deg = calc_angle_3_gps_points(gps_3points[0, :], gps_3points[1, :], gps_3points[2, :])
print(deg)
# Plot 2 lines of 3 points
plt.plot((gps_3points[0, 0], gps_3points[1, 0]), (gps_3points[0, 1], gps_3points[1, 1]), c='blue')
plt.plot((gps_3points[1, 0], gps_3points[2, 0]), (gps_3points[1, 1], gps_3points[2, 1]), c='red')
plt.show()
'''
'''
Index(['mean_speed', 'std_speed', 'min_speed', '25%_speed', '50%_speed',
       '75%_speed', 'max_speed', 'mean_speed_diff', 'std_speed_diff',
       'min_speed_diff', '25%_speed_diff', '50%_speed_diff', '75%_speed_diff',
       'max_speed_diff', 'mean_acceleration_est_1', 'std_acceleration_est_1',
       'min_acceleration_est_1', '25%_acceleration_est_1',
       '50%_acceleration_est_1', '75%_acceleration_est_1',
       'max_acceleration_est_1', 'mean_acceleration_diff',
       'std_acceleration_diff', 'min_acceleration_diff',
       '25%_acceleration_diff', '50%_acceleration_diff',
       '75%_acceleration_diff', 'max_acceleration_diff', 'trip_duration_h',
       'trip_length_km', 'trip_average_speed', 'area_rec', 'length_edge_h',
       'length_edge_v', 'idle_ratio', '90_mean_speed', '90_mean_speed_diff',
       '90_mean_acceleration_est_1', '90_mean_acceleration_diff',
       '90_std_speed', '90_std_speed_diff', '90_std_acceleration_est_1',
       '90_std_acceleration_diff', '90_min_speed', '90_min_speed_diff',
       '90_min_acceleration_est_1', '90_min_acceleration_diff', '90_max_speed',
       '90_max_speed_diff', '90_max_acceleration_est_1',
       '90_max_acceleration_diff', '135_mean_speed', '135_mean_speed_diff',
       '135_mean_acceleration_est_1', '135_mean_acceleration_diff',
       '135_std_speed', '135_std_speed_diff', '135_std_acceleration_est_1',
       '135_std_acceleration_diff', '135_min_speed', '135_min_speed_diff',
       '135_min_acceleration_est_1', '135_min_acceleration_diff',
       '135_max_speed', '135_max_speed_diff', '135_max_acceleration_est_1',
       '135_max_acceleration_diff', '180_mean_speed', '180_mean_speed_diff',
       '180_mean_acceleration_est_1', '180_mean_acceleration_diff',
       '180_std_speed', '180_std_speed_diff', '180_std_acceleration_est_1',
       '180_std_acceleration_diff', '180_min_speed', '180_min_speed_diff',
       '180_min_acceleration_est_1', '180_min_acceleration_diff',
       '180_max_speed', '180_max_speed_diff', '180_max_acceleration_est_1',
       '180_max_acceleration_diff', 'vehicle_id'],'''
