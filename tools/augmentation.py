from matplotlib import pyplot as plt
from clodsa.augmentors.augmentorFactory import createAugmentor
from clodsa.transformers.transformerFactory import transformerGenerator
from clodsa.techniques.techniqueFactory import createTechnique
import cv2

PROBLEM = "instance_segmentation"
ANNOTATION_MODE = "coco"
INPUT_PATH = "../instance_version/train"
GENERATION_MODE = "linear"
OUTPUT_MODE = "coco"
OUTPUT_PATH= "../instance_version/train_output/"
IGNORE_CLASSES = {1,12,17}
augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,INPUT_PATH,
                            {"outputPath":OUTPUT_PATH,"ignoreClasses":IGNORE_CLASSES})
							
transformer = transformerGenerator(PROBLEM)
print(transformer)

for angle in [90,180]:
    rotate = createTechnique("rotate", {"angle" : angle})
    augmentor.addTransformer(transformer(rotate))
	
flip = createTechnique("flip",{"flip":1})
augmentor.addTransformer(transformer(flip))

print(augmentor)

augmentor.applyAugmentation()	