import requests
import pandas as pd

url = "http://10.43.101.149/data"
params = {'group_number': '2'}
headers = {'accept': 'application/json'}

response = requests.get(url, params=params, headers=headers)

COLUMN_NAMES = [
    'Elevation', 
    'Aspect', 
    'Slope', 
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am',
    'Hillshade_Noon',
    'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points',
    'Wilderness_Area',
    'Soil_Type',
    'Cover_Type']

if response.status_code == 200:
    json_data = response.json()
    df = pd.DataFrame.from_dict(json_data["data"])
    df.columns = COLUMN_NAMES
    print(df)
else:
    print("Error al realizar la solicitud:", response.status_code)