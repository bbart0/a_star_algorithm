from datetime import datetime as dt, timedelta
import csv
import math

file_path = 'katedra_testing.csv'
columns = ["", "company","line","departure_time","arrival_time","start","end",
                                              "start_lat","start_lon","end_lat","end_lon"]
coordinates = dict()


time_travelled = 0
distance = 0

def get_average_speed():
    return distance/time_travelled

def adjustDate(hour: str) -> dt :
    time_format_string = "%H:%M:%S"

    i_hour = int(hour[:2])
    adjusted_hour=""
    is_time_to_be_adjusted = i_hour >= 24
    if(is_time_to_be_adjusted):
        match hour[:2]:
            case "24":
                adjusted_hour = "00"
            case "25":
                adjusted_hour = "01"
            case "26":
                adjusted_hour = "02"
            case "27":
                adjusted_hour = "03"
            case "28":
                adjusted_hour = "04"
            case "29":
                adjusted_hour = "05"
            case "30":
                adjusted_hour = "06"
    
        result = dt.strptime(adjusted_hour+hour[2:], time_format_string)
        result = result + timedelta(days=1)
    
    else:
         result = dt.strptime(hour, time_format_string)
        
    
    return result



def get_graph():
    global time_travelled
    global distance
    graph = {}
    with open(file_path,encoding="utf-8") as file:
        reader = csv.DictReader(file, fieldnames=columns) # , is the default delimiter
        
        next(reader, None)
        # O(n)
        for row in reader:
            del row[""]
            del row["company"]

            row["departure_time"] = adjustDate(row["departure_time"])
            row["arrival_time"] = adjustDate(row["arrival_time"])

            time_travelled+= (row["arrival_time"] - row["departure_time"]).total_seconds()
            
            
            if(row["start"] not in coordinates):
                 coordinates[row["start"]] = (float(row["start_lon"]),float(row["start_lat"]))
            if row["end"] not in coordinates:
                coordinates[row["end"]] = (float(row["end_lon"]),float(row["end_lat"]))

            distance+= math.sqrt((coordinates[row["end"]][0] - coordinates[row["start"]][0])**2 + (coordinates[row["end"]][1] - coordinates[row["start"]][1])**2)
            # row["start_lat"] = float(row["start_lat"])
            # row["start_lon"] = float(row["start_lon"])
            # row["end_lat"] = float(row["end_lat"])
            # row["end_lon"] = float(row["end_lon"])
            start_node = row['start']
            end_node = row['end']
            if start_node not in graph:
                graph[start_node] = {}
            if end_node not in graph:
                graph[end_node] = {}
            if end_node not in graph[start_node]:
                graph[start_node][end_node] = []
            graph[start_node][end_node].append((row['line'], row['departure_time'], row['arrival_time']))
    return graph