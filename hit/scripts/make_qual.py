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

round1_turkers = ["A2LMQ4497NMK3S", "A2RBF3IIJP15IH", "AKQAI78JTXXC9"]
datapoint = {"Worker ID": None, "UPDATE-CompletedCalibrationRound1": 100}
with open("hit/workers/worked_on_calibration_round1.csv", "w") as f1:
    writer = csv.DictWriter(f1, fieldnames=datapoint.keys()) 
    writer.writeheader() 
    for worker in round1_turkers:
        datapoint["Worker ID"] = worker 
        writer.writerow(datapoint)

for worker in all_turkers:
    worker_datapoint = {"Worker ID": worker, f"UPDATE-is_{worker}": 100}
    with open(f"hit/workers/is_{worker}.csv", "w") as f1:
        writer = csv.DictWriter(f1, fieldnames=worker_datapoint.keys()) 
        writer.writeheader() 
        writer.writerow(worker_datapoint)

