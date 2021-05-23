val_json = open(json_path, "r")
json_object = json.load(val_json)
val_json.close()

annos = json_object["annotations"]

classes = np.asarray([x["category_id"] for x in annos if not x.get("iscrowd", 0)], dtype=np.int)
histogram = np.unique(classes,return_counts=True)[1]
print(histogram)


'''
train histogram
[2633  713 1575  530  508  598  521  352  153  302  268  727  131  846
  125 1091 2203  728  320  228  210  112]
  
val histogram
[684, 200, 305, 254, 163, 163, 150, 123,  37,  86,  64, 181,  37,
       151,  16, 301, 553, 172,  97,  67,  43,  45]
'''