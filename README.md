# STV-MNet

**Abstract:**
Traditional methods for estimating the carbon storage of street trees involve manual sampling, which incurs substantial human, material, and temporal costs in establishing a comprehensive inventory of street trees across an entire city. In this study, we propose a multi-task convolutional neural network (STV-MNet) for individual-tree-level street tree identification from Baidu street view images (BSVIs). We measured the structural and locational information of the identified trees using cylindrical projection and MonoDepth depth estimation network. Experimental results show that STV-MNet achieves a mean intersection over union (mIoU) of 0.733 and a mean average precision at IoU 50% (mAP50) of 0.881 in individual tree identification, outperforming DeepLab v3+ (mIoU of 0.641) and YOLO v3 (mAP50 of 0.767). Validation with street measured data demonstrates that our method produces more precise estimations for both tree height and breast diameter, with the root mean square error (RMSE) of 1.68 m and the normalized RMSE (NRMSE) of 0.24 m for tree height and the RMSE of 0.09 m and the NRMSE of 0.29 m for chest diameter. The location prediction of street trees achieves a minimum error of 0.67 m and an average error of 7.37 m. Using the biomass carbon storage equation, we further calculated the carbon storage of each individual street tree in Changsha City, Hunan Province, China. The results indicate that the total carbon storage of 333,717 street trees in urban areas of Changsha City is 1.64 ×10<sup>5</sup> tons, and the annual carbon sequestration capacity across the urban areas is 8014.57 tons. In certain areas, the existence of street tree resources has enabled the achievement of carbon neutrality in road transportation. This study presents a novel approach for managing urban street tree carbon storage, leveraging STV-MNet for automatic carbon storage calculations, and demonstrates its high practical significance in low-cost and large-scale street tree carbon storage estimation.

**Code Description:**

The **Location** folder contains code for generating single-tree depth estimation using MonoDepth, as well as code for batch projection coordinates and tree coordinate calculation.

The **Others** folder includes code for calculating tree carbon stock, carbon density, and carbon emissions.

The **STV-MNet** folder contains the main code for model object detection and semantic segmentation, as well as training, testing, and validation code for YOLO v3 and DeepLab v3.

The **structure** folder includes R code for calculating tree height and diameter at breast height (DBH).

**Link to Paper:**
DOI:10.2139/ssrn.4974352

***

# Table of Contents
- [Environment](#installment)  
- [Acknowlegements](#Acknowlegements) 

# Environment

beautifulsoup4      4.12.2

Brotli              1.0.9

bs4                 0.0.1

certifi             2023.7.22

cffi                1.15.1

charset-normalizer  3.3.2

colorama            0.4.6

contourpy           1.2.0

cryptography        41.0.3

cycler              0.12.1

fake-useragent      1.3.0

filelock            3.13.1

fonttools           4.44.0

fsspec              2023.10.0

gitdb               4.0.11

GitPython           3.1.40

idna                3.4

Imager              0.2

importlib-resources 6.1.1

Jinja2              3.1.2

kiwisolver          1.4.5

MarkupSafe          2.1.1

matplotlib          3.8.1

mkl-fft             1.3.8

mkl-random          1.2.4

mkl-service         2.4.0

mpmath              1.3.0

networkx            3.1

numpy               1.26.0

opencv-python       4.8.1.78

packaging           23.2

pandas              2.1.2

Pillow              10.0.1

pip                 23.3

psutil              5.9.6

py-cpuinfo          9.0.0

pycocotools         2.0.7

pycparser           2.21

pyOpenSSL           23.2.0

pyparsing           3.1.1

PySocks             1.7.1

python-dateutil     2.8.2

pytz                2023.3.post1

PyYAML              6.0.1

requests            2.31.0

rpy2                3.5.14

scipy               1.11.3

seaborn             0.13.0

setuptools          68.0.0

six                 1.16.0

smmap               5.0.1

soupsieve           2.5

sympy               1.11.1

thop                0.1.1.post2209072238

torch               2.0.0

torchaudio          2.0.0

torchvision         0.15.0

tqdm                4.66.1

typing_extensions   4.7.1

tzdata              2023.3

tzlocal             5.2

ultralytics         8.0.207

urllib3             1.26.18

wheel               0.41.2

win-inet-pton       1.1.0

zipp                3.17.0

# Acknowlegements
We thank Research Professor ZHANG Lian-kai (Kunming Natural Resources Comprehensive Survey Center, China Geological Survey) for his kind assistance in data collection. The authors also would like to express their gratitude to the MapGIS Laboratory Co-Constructed by the National Engineering Research Center for Geographic Information System of China and Central South University, for providing MapGIS® software (Wuhan Zondy Cyber-Tech Co. Ltd., Wuhan, China).
