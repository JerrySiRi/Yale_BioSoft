The file briefly describes how to use clamp-api-demo jar to convert different file formats between CLAMP XMI and BIO/BRAT, and to train CRF NER model.

(1) To convert BIO to XMI files, usage:
java -cp clamp-api-demo-1.0.0-SNAPSHOT.jar ClampApi.BioToXmi -i bioPath -o xmiPath -e .bio
where -i bioPath specify the input files path, -o xmiPath specify output files path which needs to be created before run the jar, 
-e .bio specify the input BIO files extenstion, e.g., .bio or .txt. Please note that -i and -o parameters are mandotary and -e is optional, if -e is not specified, it will use the default .bio extension.

(2) To convert XMI to BIO files, usage:
java -cp clamp-api-demo-1.0.0-SNAPSHOT.jar ClampApi.XmiToBio -i xmiPath -o bioPath
where -i xmiPath specify the input files path, -o bioPath specify output files path which needs to be created before run the jar.

(3) To convert BRAT ANN to XMI files, usage:
java -cp clamp-api-demo-1.0.0-SNAPSHOT.jar ClampApi.BratToXmi -i bratPath -o xmiPath
where -i bratPath specify the input files path, -o xmiPath specify output files path which needs to be created before run the jar.
.
(4) To convert XMI to Brat ANN files, usage:
java -cp clamp-api-demo-1.0.0-SNAPSHOT.jar ClampApi.XmiToBrat -i xmiPath -o bratPath
where -i xmiPath specify the input files path, -o bratPath specify output files path which needs to be created before run the jar.

(5) To convert XMI to XMI with certain entities to be kept, usage:
java -cp clamp-api-demo-1.0.0-SNAPSHOT.jar ClampApi.XmiClean -i xmiInPath -o xmiOutPath -e ENTITY1,ENTITY2
where -i xmiInPath specify the input files path, -o xmiOutPath specify output files path which needs to be created before run the jar,
-e specify the entities that will be kept during the convertion.

(6) To train CRF NER model, usage:
java -cp clamp-api-demo-1.0.0-SNAPSHOT.jar ClampApi.NERTrain -train trainPath -test testPath -out outPath -feature featurePath
where -train trainPath specifies the train files path, -test testPath specifies the test files path, -out outPath specifies output path, and -feature speficies the absolute feature path.
The featurePath is optional; if not specified, the path NER_feature_extractor in the same folder will be used by default.