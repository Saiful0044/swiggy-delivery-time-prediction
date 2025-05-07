import numpy as np
import pandas as pd
from pathlib import Path
import logging

# create logger
logger = logging.getLogger('data_cleaning')
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

# drop column 
columns_to_drop =  ['rider_id',
                    'restaurant_latitude',
                    'restaurant_longitude',
                    'delivery_latitude',
                    'delivery_longitude',
                    'order_date',
                    "order_time_hour",
                    "order_day",
                    "city_name",
                    "order_day_of_week",
                    "order_month"]

def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error('The file to load does not exits')

    return df

# Function to rename columns to standardized names
def change_column_names(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.rename(str.lower, axis=1)
            .rename({
                "delivery_person_id": "rider_id",
                "delivery_person_age": "age",
                "delivery_person_ratings": "ratings",
                "delivery_location_latitude": "delivery_latitude",
                "delivery_location_longitude": "delivery_longitude",
                "time_orderd": "order_time",
                "time_order_picked": "order_picked_time",
                "weatherconditions": "weather",
                "road_traffic_density": "traffic",
                "city": "city_type",
                "time_taken(min)": "time_taken"
            }, axis=1)
    )

# Function to categorize hours into parts of the day
def time_of_day(hour_series):
    return(
        pd.cut(hour_series, bins=[0,6,12,17,20,24],right=True,
               labels=["after_midnight","morning","afternoon","evening","night"])
    )


# Main data cleaning function
def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    # Filter out underage riders and invalid ratings
    minors_index = data.loc[data['age'].astype('float') < 18].index
    six_star_index = data.loc[data['ratings'] == '6'].index

    return (
        data
        .drop(columns='id')                         # Remove unused ID column
        .drop(index=minors_index)                   # Drop minors
        .drop(index=six_star_index)                 # Drop invalid 6-star ratings
        .replace('NaN ', np.nan)                    # Replace incorrect NaN strings

        # Feature engineering and type conversion
        .assign(
            city_name=lambda x: x['rider_id'].str.split('RES').str.get(0),
            age=lambda x: x['age'].astype(float),
            ratings=lambda x: x['ratings'].astype(float),

            # Absolute values for lat/long (removing negatives if any)
            restaurant_latitude=lambda x: x['restaurant_latitude'].abs(),
            restaurant_longitude=lambda x: x['restaurant_longitude'].abs(),
            delivery_latitude=lambda x: x['delivery_latitude'].abs(),
            delivery_longitude=lambda x: x['delivery_longitude'].abs(),

            # Order datetime and date-based features
            order_date=lambda x: pd.to_datetime(x['order_date'], dayfirst=True),
            order_day=lambda x: x['order_date'].dt.day,
            order_month=lambda x: x['order_date'].dt.month,
            order_day_of_week=lambda x: x['order_date'].dt.day_name().str.lower(),
            is_weekend=lambda x: x['order_date'].dt.day_name().isin(['Saturday', 'Sunday']).astype(int),

            # Order and pickup times
            order_time=lambda x: pd.to_datetime(x['order_time'], format='mixed'),
            order_picked_time=lambda x: pd.to_datetime(x['order_picked_time'], format='mixed'),
            pickup_time_minutes=lambda x: (x['order_picked_time'] - x['order_time']).dt.total_seconds() / 60,

            # Hour of day and categorical time of day
            order_time_hour=lambda x: x['order_time'].dt.hour,
            order_time_of_day=lambda x: time_of_day(x['order_time'].dt.hour),

            # Clean categorical fields
            traffic=lambda x: x['traffic'].str.strip().str.lower(),
            type_of_order=lambda x: x['type_of_order'].str.strip().str.lower(),
            type_of_vehicle=lambda x: x['type_of_vehicle'].str.strip().str.lower(),
            festival=lambda x: x['festival'].str.strip().str.lower(),
            city_type=lambda x: x['city_type'].str.strip().str.lower(),

            # Convert delivery count to float
            multiple_deliveries=lambda x: x['multiple_deliveries'].astype(float),

            # Clean and convert target variable
            time_taken=lambda x: x['time_taken'].str.replace('(min) ', '', regex=False).astype(int)
        )

        # Drop unneeded datetime columns
        .drop(columns=['order_time', 'order_picked_time'])
    )


def clean_lat_long(data: pd.DataFrame, threshold: float=1.0) -> pd.DataFrame:
    # List of columns containing location data
    location_columns = ['restaurant_latitude', 'restaurant_longitude',
                       'delivery_latitude', 'delivery_longitude']

    return (
        data.assign(**{  # Creates new columns/modifies existing ones
            col: (  # For each location column:
                np.where(data[col] < threshold,  # Condition: value < threshold
                        np.nan,                 # If True: replace with NaN
                        data[col])              # If False: keep original value
            ) for col in location_columns  # Loop through all location columns
        })
    )

# extract day, day name, month and year
def extract_datetime_features(hour_series: pd.Series) -> pd.DataFrame:
    # Convert input series to proper datetime format (day-first)
    date_col = pd.to_datetime(hour_series, dayfirst=True)

    return (
        pd.DataFrame({  # Create new DataFrame with extracted features
            'day': date_col.dt.day,          # Day of month (1-31)
            'month': date_col.dt.month,      # Month (1-12)
            'year': date_col.dt.year,        # Year (e.g., 2023)
            'day_of_week': date_col.dt.day_name(),  # Weekday name (Monday-Sunday)
            'is_weekend': date_col.dt.day_name().isin(['Saturday','Sunday']).astype(int)
            # Binary flag (1=weekend, 0=weekday)
        })
    )

def calculate_haversine_distance(df) -> pd.DataFrame:
    # List of column names containing location coordinates
    location_columns = [
        'restaurant_latitude',          'restaurant_longitude',
        'delivery_latitude',
        'delivery_longitude'
    ]

    # Extract coordinate values from DataFrame
    lat1 = df[location_columns[0]]  # Restaurant latitude
    lon1 = df[location_columns[1]]  # Restaurant longitude
    lat2 = df[location_columns[2]]  # Delivery latitude (FIXED: was [3])
    lon2 = df[location_columns[3]]  # Delivery longitude (FIXED: was [4])

    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Calculate differences
    dlon = lon2 - lon1  # Difference in longitude
    dlat = lat2 - lat1  # Difference in latitude

    # Haversine formula components
    a = (np.sin(dlat/2.0)**2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2)

    # Calculate angular distance in radians
    c = 2 * np.arcsin(np.sqrt(a))

    # Convert to kilometers (Earth's radius = 6371 km)
    distance = 6371 * c

    # Return new DataFrame with distance column added
    return df.assign(distance=distance)


def create_distance_type(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.assign(  # Returns a new DataFrame with added column
            distance_type = pd.cut(
                data['distance'],  # Column to bin
                bins=[0, 5, 10, 15, 25],  # Bin edges
                right=False,  # Interval inclusion (important!)
                labels=['short', 'medium', 'long', 'very_long']  # Category names
            )
        )
    )

# drop columns
def drop_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = data.drop(columns=columns)
    return df

def perform_data_cleaning(data: pd.DataFrame, saved_data_path: Path) -> None:
    cleaned_data = (
        data
        .pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        .pipe(create_distance_type)
        .pipe(drop_columns,columns=columns_to_drop)
    )

    # save the data
    cleaned_data.to_csv(saved_data_path, index=False)


if __name__=="__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # data save directory
    cleaned_data_save_dir = root_path / "data" / "cleaned"
    # make directory if not exits
    cleaned_data_save_dir.mkdir(exist_ok=True,parents=True)
    # cleaned data file name
    cleaned_data_filename = 'swiggy_cleaned.csv'
    # data save path 
    cleaned_data_save_path = cleaned_data_save_dir / cleaned_data_filename
    # data load path
    data_load_path = root_path / "data" / "raw" / "swiggy.csv"

    # load the data
    df = load_data(data_load_path)
    logger.info("Data read successfully")

    # clean the data and save
    perform_data_cleaning(data=df, saved_data_path=cleaned_data_save_path)
    logger.info("Data cleaned and saved")
