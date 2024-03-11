def total_euro(working_hours, euro_per_hour):
    return working_hours * euro_per_hour


working_hours = input("Radni sati: ")
working_hours = working_hours.split(" ")[0]
working_hours = int(working_hours)

euro_per_hour = input("eura/h: ")
euro_per_hour = float(euro_per_hour)

print(f"Ukupno: {total_euro(working_hours, euro_per_hour)} eura")
