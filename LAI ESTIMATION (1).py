#!/usr/bin/env python
# coding: utf-8

# In[23]:


#MODULE NAME
import os
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.exposure as exposure
import pandas as pd
import numpy as np
import subprocess
import snappy
import imageio
import rasterio as rio
import rasterstats as rs
import rasterio.plot
from glob import iglob
import geopandas as gpd
from pyspatialml import Raster
import seaborn as sns
from rasterio.plot import show
get_ipython().run_line_magic('matplotlib', 'inline')
from osgeo import gdal


# # Vegetation Indices

# In[24]:


with rasterio.open(r'C:\Users\BSibiya\Desktop\LAI ESTIMATION\Image\sentinel_image2.tif') as src:
    blue = src.read(1, masked=True) #B2
    green = src.read(2, masked=True) #B3
    red = src.read(3, masked=True) #B4
    Red_Edge_1 = src.read(4, masked=True) #B5
    Red_Edge_2 = src.read(5, masked=True) #B6
    Red_Edge_3 = src.read(6, masked=True) #B7
    nir = src.read(7, masked=True) #B8
    swir_1 = src.read(8, masked=True) #B11
    swir_2 = src.read(9, masked=True) #B12
    
'B2','B3','B4','B5','B6','B7','B8','B11','B12    
    
np.seterr(divide='ignore', invalid='ignore')
VI_output = '/Users/BSibiya/Desktop/LAI ESTIMATION/VI_Output/'


# # User-defined function

# In[25]:


#Function to calculate vegetation indices from band combinations
#Raster_output_name - name of the vegetation index which will be saved 
#VI_name - name of the vegetation index
#cmap_output_name - name of the png file where the file will be stored with colormap applied
#plot_title - title of the image produced 

def create_vegetation_indices(raster_output_name, VI_name, cmap_output_name, plot_title):
    np.seterr(divide='ignore', invalid='ignore')
    kwargs = src.meta
    kwargs.update(
        dtype=rasterio.float32,
        count = 1)
    
    with rasterio.open(VI_output + raster_output_name, 'w', **kwargs) as dst:
        dst.write_band(1, VI_name.astype(rasterio.float32))
    
    plt.imsave(VI_output + cmap_output_name, VI_name, cmap=plt.cm.RdYlGn)
    plt.imshow(VI_name, cmap=cm.RdYlGn)
    plt.colorbar()
    plt.title(plot_title)
    plt.show()
    
#Normalizes numpy arrays into scale -1.0 - 1.0
#array - indicates in this case the array of the image to normalize
def normalize(array):
    array_min, array_max = array.min(-1), array.max(1)
    return ((array - array_min)/(array_max - array_min))


# # Normalized Difference vegetation Index - NDVI

# In[26]:


#Equation to calculate NDVI
ndvi = (nir.astype(float) - red.astype(float)) / (nir + red)
#create NDVI bands with the use of create_vegetation_indices function from section 3 - user defined functions 
#The result will be written into tiff and png file with the colormap applied
#ndvi_results = create_vegetation_indices('1_NDVI.tif', ndvi, '1_NDVI_cmap.png', 'NDVI_results')


# # Normalized Difference index - NDI45

# In[27]:


#Equation to calculate NDI45
ndi45 = (Red_Edge_1.astype(float) - red.astype(float)) / (Red_Edge_1 + red)
#create NDI45 bands with the use of create_vegetation_indices function from section 3 - user defined functions 
#The result will be written into tiff and png file with the colormap applied
#ndi45_results = create_vegetation_indices('2_NDI45.tif', ndi45, '2_NDI45_cmap.png', 'NDI45_results')


# # Soil Adjusted Vegetation Index - SAVI

# In[28]:


#Calculate SAVI 
l = 0.428 #L parameter assigned
savi = (nir.astype(float) - red.astype(float)) / (nir + red + l) * (1 + l)
#normalize raster results to scale: -1 tp 1:
#savi_n = normalize(savi)
#create SAVI band with the use of create_vegetation_indices function from section 3 - user-defined function
#savi_results = create_vegetation_indices('3_SAVI.tif', savi, '3_SAVI_cmap.png', 'SAVI_results')


# # Normalized Difference Moisture Index - NDMI

# In[29]:


#Normalized difference Moisture Index - NDMI
ndmi = (nir.astype(float) - swir_1.astype(float)) / (nir + swir_1)
#create NDMI band with the use of create_vegetation_indices function from section 3 - user-defined function
#ndmi_results = create_vegetation_indices('4_NDMI.tif', savi, '4_NDMI_cmap.png', 'NDMI_results')


# # RASTER STACK OF VEGETATION INDICES

# In[33]:


#import list of files (rasters) to create image stack 
from glob import glob
glist = sorted(glob('/Users/BSibiya/Desktop/LAI ESTIMATION/VI_Output/*.tif')) #import list of products with the extension .tif
glist #print list of products which will be included in raster stack 


# In[34]:


#read all metadata of all single raster bands:
with rasterio.open(glist[0]) as src0:
    meta = src0.meta
meta.update(count = len(glist))
#rad each single band of vegetation indices and load them to rasterio module
with rasterio.open('/Users/BSibiya/Desktop/LAI ESTIMATION/VI_Output/VIs_stack.tif', 'w', **meta) as dst:
    for id, layer in enumerate(glist, start=1):
        with rasterio.open(layer) as src1:
            dst.write_band(id, src1.read(1))
#save raster stack in the output folder
with rio.open('/Users/BSibiya/Desktop/LAI ESTIMATION/VI_Output/VIs_stack.tif') as stack_src:
    stack_data = stack_src.read(masked=True)
    stack_meta = stack_src.profile
stack_meta #display metadata of the product


# # IMPORT TRAINING POINTS SHAPEFILE

# In[32]:


point_path = '/Users/BSibiya/Desktop/LAI ESTIMATION/LAI/leafAreaIndex.shp'


# In[35]:


#READ SHAPEFILE INTO PYTHON ENVIRONMENT 
pts = gpd.read_file(point_path) #points will be read into python using geopandas module 
#pts = pts.to_crs(epsg=3857)
print(pts.count()) #here we will display basic information about the points shapefile
print(pts.crs)

#IMPORT VEGETATION RASTER STACK USING RASTERIO PACKAGE 
VIs_stack = rasterio.open('/Users/BSibiya/Desktop/LAI ESTIMATION/VI_Output/VIs_stack.tif') #set path to the raster stack 
fig, ax = plt.subplots(figsize=(10,10)) #set the size of the plot 
pts.plot(ax=ax, color = 'orangered') #plot imported points and set their color
show(VIs_stack, ax=ax) #display the image (raster stack) together with points 


# # Create dataframes to store values of extracted pixel

# In[42]:


dataframe = '/Users/BSibiya/Desktop/LAI ESTIMATION/dataset.xlsx'
df = pd.read_excel(dataframe)
df.head()


# # Correlation Matrix

# In[43]:


#Visual representation of correlation matrix using seaborn python module
corrMatrix = df.corr() #create first simple correlation matrix
sns.heatmap(corrMatrix, annot=True) #plot the correlation matrix as a heatmap 
plt.show()


# # Linear Relationship between VIs and LAI values

# In[44]:


#we will plot each vegetation index against LAI values in order to see the relationship between these values 
df.plot.scatter(x='NDVI', y='Lai', s=60, c='green')
df.plot.scatter(x='NDI45', y='Lai', s=60, c='blue')
df.plot.scatter(x='SAVI', y='Lai', s=60, c='red')
df.plot.scatter(x='NDMI', y='Lai', s=60, c='orange')


# # LAI estimation using Machine Learning Techniques

# # Multiple Linear Regression Model

# In[47]:


from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import r2_score


# In[50]:


#Choose columns which include all variables (X- Predictors -VIs and y - Dependent variables) which will be used as an input to all regression model
#Use columns nanes from dataframe created previously to select predictor variables for the model
X = df.iloc[:, 1:].values #As predictors we are going to use following values" 'NDVI', 'NDI45', 'SAVI', NDMI
y = df.iloc[:,0].values #Dependent variable - LAI is stored in the first column of the dataframe 


# In[78]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)#30% of the dataset will be left to test dataset
from sklearn.linear_model import LinearRegression
mlr_regressor = LinearRegression()
mlr_regressor.fit(X_train, y_train)
y_pred = mlr_regressor.predict(X_train)
r2_score(y_train, y_pred)
#Write Calculation of the results to seperate Dataframe and display:
mlr_result = {'Intercept': [mlr_regressor.intercept_],
              'Coefficients': [mlr_regressor.coef_],
              'r-squared score': [mlr_regressor.score(X_train, y_train)]}

mlr_df = pd.DataFrame(mlr_result, columns = ['Intercept', 'Coefficients', 'r-squared score'])
mlr_df



# # Random Forest Regressor

# In[52]:


X_data = df.iloc[:, 1:]
y_data = df.iloc[:,0]


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.30, random_state = 0)
from sklearn.ensemble import RandomForestRegressor
RFReg = RandomForestRegressor(n_estimators=100, max_depth=3, n_jobs=-1, random_state=0)
RFReg.fit(X_train, y_train)
y_pred_RFReg = RFReg.predict(X_train)
from sklearn import metrics
r_square_rf = metrics.r2_score(y_train, y_pred_RFReg)
print('r-square: ', r_square_rf)


# In[55]:


#The values of the importance will be calculated based on the model
rf_features = df.iloc[:, 1:]
features_list = list(rf_features.columns)
importances = list(RFReg.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features_list, importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# # Create Geotiff function

# In[126]:


def createGeotiff(outRaster, data, geo_transform, projection):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    rasterDS = driver.Create(outRaster, cols, rows, 1, gdal.GDT_Float32)
    rasterDS.SetGeoTransform(geo_transform)
    rasterDS.SetProjection(projection)
    band = rasterDS.GetRasterBand(1)
    band.WriteArray(data)
    rasterDS = None 


# # Create Leaf Area Index Maps

# #  Leaf Area Index based on Linear Regression Model

# In[157]:


#Here we create the geotiff file which will contain predicted LAI values 
# Load the predictor variables (VIs_stack)

inRaster = '/Users/BSibiya/Desktop/LAI ESTIMATION/VI_Output/VIs_stack.tif' #Define path to the input raster
    

df_train = pd.read_excel(dataframe) #path to the dataset with training data
data = df_train[['NDVI', 'NDI45', 'SAVI', 'NDMI']] #list of predictors 
label = df_train['Lai'] #predicted variable(dependent variable LAI)
ds = gdal.Open(inRaster, gdal.GA_ReadOnly) #with gdal module we will open the input raster
rows = ds.RasterYSize #we need to assign the size of the raster - rows number in the raster dataset
cols = ds.RasterXSize #we need to assign the size of the raster - columns number in the raster dataset
bands = ds.RasterCount # we will also get the number of bands in the input raster (in our case 4)
geo_transform = ds.GetGeoTransform() #with geotransform class we will set projection to the raster
projection = ds.GetProjectionRef() #projection of the raster needs to be the same as input raster
array = ds.ReadAsArray() #raster will be converted to an array 
ds = None
array = np.stack(array, axis=2) #with numpy we will join the sequence of the arrays along a new axes
array = np.reshape(array, [rows*cols, bands]) #we give a final shape to the output raster: number of columns and rows and bands must be the same as 
test = pd.DataFrame(array, dtype='float32') #we will create raster with the same datatype 'float32' as input raster
outRaster = '/Users/BSibiya/Desktop/LAI ESTIMATION/VI_Output/LAI_mlr.tif' #path to the output raster - where it will be saved 
#PREDICTION OF ABOVEGROUND BIOMASS VALUES BASED ON THE TRAINING DATASET AND THE RASTER STACK PROVIDED
LAI_mlr = mlr_regressor.predict(test) #run the prediction on multiple linear regression model previously created 
estimation = LAI_mlr.reshape((rows, cols)) #reshape output array into array with same defined previously number of columns and rows



# In[158]:


#explot classified image using 'createGeoftiff' function provided in the section number 3
createGeotiff(outRaster, estimation, geo_transform, projection)


# # Leaf Area Index based on Random Forest Regression Model

# In[159]:


#Here we create the geotiff file which will contain predicted LAI values 
# Load the predictor variables (VIs_stack)

inRaster = '/Users/BSibiya/Desktop/LAI ESTIMATION/VI_Output/VIs_stack.tif' #Define path to the input raster
    

df_train = pd.read_excel(dataframe) #path to the dataset with training data
data = df_train[['NDVI', 'NDI45', 'SAVI', 'NDMI']] #list of predictors 
label = df_train['Lai'] #predicted variable(dependent variable LAI)
ds = gdal.Open(inRaster, gdal.GA_ReadOnly) #with gdal module we will open the input raster
rows = ds.RasterYSize #we need to assign the size of the raster - rows number in the raster dataset
cols = ds.RasterXSize #we need to assign the size of the raster - columns number in the raster dataset
bands = ds.RasterCount # we will also get the number of bands in the input raster (in our case 4)
geo_transform = ds.GetGeoTransform() #with geotransform class we will set projection to the raster
projection = ds.GetProjectionRef() #projection of the raster needs to be the same as input raster
array = ds.ReadAsArray() #raster will be converted to an array 
ds = None
array = np.stack(array, axis=2) #with numpy we will join the sequence of the arrays along a new axes
array = np.reshape(array, [rows*cols, bands]) #we give a final shape to the output raster: number of columns and rows and bands must be the same as 
test1 = pd.DataFrame(array, dtype='float32') #we will create raster with the same datatype 'float32' as input raster
outRaster = '/Users/BSibiya/Desktop/LAI ESTIMATION/VI_Output/LAI_RFReg1.tif' #path to the output raster - where it will be saved 
#PREDICTION OF ABOVEGROUND BIOMASS VALUES BASED ON THE TRAINING DATASET AND THE RASTER STACK PROVIDED
LAI_RFReg = RFReg.predict(test1) #run the prediction on multiple linear regression model previously created 
estimation = LAI_RFReg.reshape((rows, cols)) #reshape output array into array with same defined previously number of columns and rows



# In[160]:


#explot classified image using 'createGeoftiff' function provided in the section number 3
createGeotiff(outRaster, estimation, geo_transform, projection)


# # Visualization of Leaf Area Index maps Produced 

# In[161]:


#read the output raster and open it 
LAI_mlr = rasterio.open(r'C:\Users\BSibiya\Desktop\LAI ESTIMATION\VI_Output\LAI_mlr.tif')
LAI_mlr_array = LAI_mlr.read() #read all bands as an array
#calculate statistics for the image 
stats = []
for band in LAI_mlr_array:
    stats.append({
        'min': band.min(),
        'mean': band.mean(),
        'median': np.median(band),
        'max': band.max()
    })
print(stats)


# In[162]:


#read the output raster and open it 
LAI_RF = rasterio.open(r'C:\Users\BSibiya\Desktop\LAI ESTIMATION\VI_Output\LAI_RFReg1.tif')
LAI_RF_array = LAI_mlr.read() #read all bands as an array
#calculate statistics for the image 
stats = []
for band in LAI_mlr_array:
    stats.append({
        'min': band.min(),
        'mean': band.mean(),
        'median': np.median(band),
        'max': band.max()
    })
print(stats)


# In[163]:


show(LAI_mlr)


# In[164]:


show(LAI_RF)


# In[ ]:




