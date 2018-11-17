import csv
import numpy as np
from sklearn import preprocessing

def loadCSV(filename, city):
    print("======== Load CSV File ========")
    records = []
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row.
        for row in reader:
            totalPM25 = -1
            stations = 0
            for pm25 in row[6:-8]:                
                if pm25 != "NA":
                    if totalPM25 == -1:
                        totalPM25 = 0                    
                    stations += 1
                    totalPM25 += int(pm25)
            if stations != 0 and "NA" not in row[:6] and "NA" not in row[-9:] and row[-8] != "-9999" and row[-7] != "-9999":
                row[-4] = 0 if row[-4] == "NE" else 1 if row[-4] == "NW" else 2 if row[-4] == "SE" else 3 if row[-4] == "SW" else 4
                record = row[1:6] + row[-8:]
                record.append(city)
                record.append(totalPM25 / stations)
                records.append(record)    
    return list(map(list, zip(*records)))

def normalizeData(records):
    records[:5] = np.array(records[:5]).astype(int)
    records[9] = np.array(records[9]).astype(int)
    records[13] = np.array(records[13]).astype(int)
    records[14] = np.array(records[14]).astype(float)
    cols = [5, 6, 7, 8, 10, 11, 12]
    for col in cols:
        records[col] = preprocessing.minmax_scale(records[col], (0, 1), 0)
    return records

def writeCSV(records):
    transformedRecords = list(map(list, zip(*records)))
    with open('allPMData.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Year", "Month", "Day", "Hour", "Season", "DEWP", 
            "HWMI", "PRES", "TEMP", "CBWD", "IWS", "RAIN", "IPREC", "City", "AVG PM2.5"])
        for record in transformedRecords:
            writer.writerow(record)

def concateAllData(allRecords, appendRecords):
    for i in range(len(allRecords)):
        allRecords[i].extend(appendRecords[i])

if __name__ == "__main__":
    allRecords = []
    allRecords.extend(loadCSV('data/Beijing.csv', 0))   # Bejing: 0
    concateAllData(allRecords, loadCSV('data/Chengdu.csv', 1))   # Chengdu: 1
    concateAllData(allRecords, loadCSV('data/Guangzhou.csv', 2))   # Guangzhou: 2
    concateAllData(allRecords, loadCSV('data/Shanghai.csv', 3))   # Shanghai: 3
    concateAllData(allRecords, loadCSV('data/Shenyang.csv', 4))   # Shenyang: 4
    
    allRecords = normalizeData(allRecords)
    writeCSV(allRecords)
    