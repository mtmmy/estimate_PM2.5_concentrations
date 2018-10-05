import csv

class Record:
    def __init__(self, data, avgPM25, city):
        self.year = int(data[1])
        self.month = int(data[2])
        self.day = int(data[3])
        self.hour = int(data[4])
        self.season = int(data[5])
        self.iprec = float(data[-1])
        self.rain = float(data[-2])
        self.iws = float(data[-3])        
        self.temp = float(data[-5])
        self.pres = float(data[-6])
        self.hwmi = float(data[-7])
        self.dewp = float(data[-8])
        self.city = city
        self.avgPM25 = avgPM25
        self.cbwd = 0 if data[-4] == "NE" else 1 if data[-4] == "NW" else 2 if data[-4] == "SE" else 3 if data[-4] == "SW" else 4
        # NE: 0; NW: 1; SE: 2; SW: 3; ca: 4
    
    def getRow(self):
        return [self.year, self.month, self.day, self.hour, self.season, self.dewp, self.hwmi, 
        self.pres, self.temp, self.cbwd, self.iws, self.rain, self.iprec, self.city, self.avgPM25]

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
            if stations != 0 and "NA" not in row[:6] and "NA" not in row[-9:]:
                records.append(Record(row, totalPM25 / stations, city))
    
    return records

def writeCSV(records):
    with open('allPMData.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Year", "Month", "Day", "Hour", "Season", "DEWP", 
            "HWMI", "PRES", "TEMP", "CBWD", "IWS", "RAIN", "IPREC", "City", "AVG PM2.5"])
        for record in records:
            writer.writerow(record.getRow())

if __name__ == "__main__":
    allRecords = []
    allRecords.extend(loadCSV('data/Beijing.csv', 0))   # Bejing: 0
    allRecords.extend(loadCSV('data/Chengdu.csv', 1))   # Chengdu: 1
    allRecords.extend(loadCSV('data/Guangzhou.csv', 2))   # Guangzhou: 2
    allRecords.extend(loadCSV('data/Shanghai.csv', 3))   # Shanghai: 3
    allRecords.extend(loadCSV('data/Shenyang.csv', 4))   # Shenyang: 4
    
    writeCSV(allRecords)
    