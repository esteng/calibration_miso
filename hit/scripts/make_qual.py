import csv 
import pathlib 

all_turkers = ["A28AXX4NCWPH1F", "A2LMQ4497NMK3S", "A2RBF3IIJP15IH", "AKQAI78JTXXC9"]


datapoint = {"Worker ID": None, "UPDATE-CompletedCalibrationPilot": 100}
to_write = []
with open("hit/workers/worked_on_calibration.csv", "w") as f1:
    writer = csv.DictWriter(f1, fieldnames=datapoint.keys()) 
    writer.writeheader() 
    for worker in all_turkers:
        datapoint["Worker ID"] = worker 
        writer.writerow(datapoint)




