Data Used in the Experiment:
The data used in this experiment is stored in the deeplearning folder.
The experiment code and modifications follow the workflow of a blogger on CSDN, and the code is from an open-source project on GitHub. Link to the CSDN blog

Environment Dependencies:
The experiment was conducted on the PyCharm platform, with a computer featuring an NVIDIA GTX1650 GPU that supports CUDA acceleration. Detailed installation instructions can be found here: https://blog.csdn.net/chen565884393/article/details/127905428.
Note: Be sure to install Visual Studio before proceeding with the installation.
For installing PyTorch, follow the guide here: https://blog.csdn.net/MCYZSF/article/details/116525159

Environment:
appdirs: 1.4.4
brotlipy: 0.7.0
cachetools: 5.3.0
certifi: 2022.12.7
cffi: 1.15.1
charset-normalizer: 2.0.4
colorama: 0.4.6
contourpy: 1.0.5
cryptography: 39.0.1
cycler: 0.11.0
filelock: 3.9.0
flit_core: 3.8.0
fonttools: 4.25.0
google-auth: 2.17.1
google-auth-oauthlib: 1.0.0
grpcio: 1.53.0
idna: 3.4
imgviz: 1.7.2
importlib-metadata: 6.1.0
importlib-resources: 5.2.0
Jinja2: 3.1.2
kiwisolver: 1.4.4
labelme: 3.16.7
Markdown: 3.4.3
MarkupSafe: 2.1.1
matplotlib: 3.7.1
mkl-fft: 1.3.1
mkl-random: 1.2.2
mkl-service: 2.4.0
mpmath: 1.2.1
munkres: 1.1.4
natsort: 8.3.1
networkx: 2.8.4
numpy: 1.23.5
oauthlib: 3.2.2
onnx: 1.13.0
opencv-python: 4.7.0.72
packaging: 23.0
Pillow: 9.4.0
pip: 23.0.1
ply: 3.11
pooch: 1.4.0
protobuf: 3.20.3
pyasn1: 0.4.8
pyasn1-modules: 0.2.8
pycocotools: 2.0.6
pycparser: 2.21
pyOpenSSL: 23.0.0
pyparsing: 3.0.9
PyQt5: 5.15.7
PyQt5-sip: 12.11.0
PySocks: 1.7.1
python-dateutil: 2.8.2
PyYAML: 6.0
QtPy: 2.3.1
requests: 2.28.1
requests-oauthlib: 1.3.1
rsa: 4.9
scipy: 1.10.0
setuptools: 65.6.3
sip: 6.6.2
six: 1.16.0
sympy: 1.11.1
tensorboard: 2.12.1
tensorboard-data-server: 0.7.0
tensorboard-plugin-wit: 1.8.1
tensorboardX: 2.6
termcolor: 2.2.0
toml: 0.10.2
torch: 2.0.0
torchaudio: 2.0.0
torchvision: 0.15.0
tornado: 6.2
tqdm: 4.65.0
typing_extensions: 4.4.0
urllib3: 1.26.15
Werkzeug: 2.2.3
wheel: 0.38.4
win-inet-pton: 1.1.0
wincertstore: 0.2
zipp: 3.11.0

Data Processing:
After labeling with LabelMe, place the original images and the corresponding JSON files into a folder (the "before" folder under the datasets folder in the code). Run the json_to_dataset.py file to generate semantic segmentation label images. The original images will be in the JPEGImages folder, and the labels will be in the SegmentationClass folder under the datasets folder.

Next, place the generated JPEGImages and SegmentationClass folders into the VOCdevkit folder of the project (this folder already contains a newly created ImageSets folder with a Segmentation subfolder). Run the voc_annotation.py file to split the data into training, validation, and test sets (the ratio can be adjusted; details are commented in the code. In the project I am using, we do not use a test set and treat the validation set as the test set).

If data augmentation is required, place the images in the image folder of the project, and run the inhance.py file. The augmented images will be saved with a "T" prefix in the image_1 folder. There is no need to perform additional LabelMe annotation for the augmented data. Simply augment the original image's semantic segmentation label data, which has been tested to work very well. (In this experiment, only horizontal flipping was used for data augmentation.)

The addTest.py file is used to add newly acquired road images to the test set.






