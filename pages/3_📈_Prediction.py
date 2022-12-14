from math import ceil, floor
import pickle
import numpy as np
import streamlit as st
import pandas as pd

def app():
    df = pd.read_csv(r"data/house_price.csv")
    df.rename(columns={'LotArea': 'Lot size square feet', 'GrLivArea': 'Above grade (ground) living area square feet', 'BsmtFinSF1':'Type 1 finished square feet', 'LotFrontage': 'Linear feet of street connected to property', 'TotalBsmtSF': 'Total square feet of basement area', 'Neighborhood': 'Physical locations within Ames city limits', '1stFlrSF': 'First Floor square feet', 'GarageArea': 'Size of garage square feet', 'MasVnrArea': 'Masonry veneer area square feet', 'OverallQual': 'Overall material and finish quality', 'YearRemodAdd': 'Remodel date', '2ndFlrSF': 'Second floor square feet', 'MasVnrAreaCatg': 'Masonry veneer area category', 'GarageYrBlt': 'Year garage was built', 'YearBuilt': 'Original construction date', 'Fireplaces':'Number of fireplaces', 'TotRmsAbvGrd': 'Total rooms above grade (does not include bathrooms)', 'ExterQual': 'Exterior material quality', 'GarageCars': 'Size of garage in car capacity', 'HouseStyle': 'Style of housing'}, inplace=True)

    dropColumns = ["Id", "MSSubClass", "MSZoning", "Street", "LandContour", "Utilities", "LandSlope", "Condition1", "Condition2", "BldgType", "OverallCond", "RoofStyle",
                "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterCond", "Foundation", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "Heating", "Electrical", "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "HalfBath"] + ["SaleCondition", "SaleType", "YrSold", "MoSold", "MiscVal", "MiscFeature", "Fence", "PoolQC", "PoolArea", "ScreenPorch", "3SsnPorch", "EnclosedPorch", "OpenPorchSF", "WoodDeckSF", "PavedDrive", "GarageCond", "GarageQual", "GarageType", "FireplaceQu", "Functional", "KitchenAbvGr", "BedroomAbvGr"]

    droppedDf = df.drop(columns=dropColumns, axis=1)

    droppedDf.isnull().sum().sort_values(ascending=False)
    droppedDf["Alley"].fillna("NO", inplace=True)
    droppedDf["Linear feet of street connected to property"].fillna(df["Linear feet of street connected to property"].mean(), inplace=True)
    droppedDf["GarageFinish"].fillna("NO", inplace=True)
    droppedDf["Year garage was built"].fillna(df["Year garage was built"].mean(), inplace=True)
    droppedDf["BsmtQual"].fillna("NO", inplace=True)
    droppedDf["Masonry veneer area square feet"].fillna(0, inplace=True)
    droppedDf['Masonry veneer area category'] = np.where(droppedDf["Masonry veneer area square feet"] > 1000, 'BIG',
                                    np.where(droppedDf["Masonry veneer area square feet"] > 500, 'MEDIUM',
                                    np.where(droppedDf["Masonry veneer area square feet"] > 0, 'SMALL', 'NO')))

    droppedDf = droppedDf.drop(['SalePrice'], axis=1)
    inputDf = droppedDf.iloc[[0]].copy()

    for i in inputDf:
        if inputDf[i].dtype == "object":
            inputDf[i] = droppedDf[i].mode()[0]
        elif inputDf[i].dtype == "int64" or inputDf[i].dtype == "float64":
            inputDf[i] = droppedDf[i].mean()

    obj_feat = list(inputDf.loc[:, inputDf.dtypes == 'object'].columns.values)
    for feature in obj_feat:
        inputDf[feature] = inputDf[feature].astype('category')

    # load the model weights and predict the target
    modelName = r"finalized_model.model"
    loaded_model = pickle.load(open(modelName, 'rb'))

    # %% STREAMLIT FRONTEND DEVELOPMENT
    st.title("House Prices Prediction")

    st.sidebar.title("Model Parameters")
    
    expander= st.sidebar
    
    # Get Feature importance of model
    featureImportances = pd.Series(loaded_model.feature_importances_,index = droppedDf.columns).sort_values(ascending=False)[:20]
    
    inputDict = dict(inputDf)

    for idx, i in enumerate(featureImportances.index):
        if droppedDf[i].dtype == "object":
            variables = droppedDf[i].drop_duplicates().to_list()
            inputDict[i] = expander.selectbox(i, options=variables, key=idx)
        elif droppedDf[i].dtype == "int64" or droppedDf[i].dtype == "float64":
            inputDict[i] = expander.slider(i, ceil(droppedDf[i].min()),
                                                floor(droppedDf[i].max()), int(droppedDf[i].mean()), key=idx)
        else:
            expander.write(i)


    for key, value in inputDict.items():
        inputDf[key] = value

    obj_feat = list(inputDf.loc[:, inputDf.dtypes == 'object'].columns.values)
    for feature in obj_feat:
        inputDf[feature] = inputDf[feature].astype('category')

    prediction = loaded_model.predict(inputDf)
    
    st.write("## $", prediction.item())
    
st.set_page_config(page_title="Prediction")

app()
