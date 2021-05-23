val_json = open(json_path, "r")
json_object = json.load(val_json)
val_json.close()
for i, instance in enumerate(json_object["annotations"]):
	if len(instance["segmentation"][0]) == 4:
			print("instance number", i, "raises arror:", instance["segmentation"][0])

#instance number 3471 raises arror: [17, 0, 17, 0]

json_object["annotations"][3471]["segmentation"] = [[17, 0, 18,0 17, 0]]
val_json = open(JSON_LOC, "w")
json.dump(json_object, val_json)
val_json.close()