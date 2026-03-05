{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "adb18395",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "# Load data and train a simple model (Vehicle_Age → Price_USD)\n",
    "data = pd.read_csv(r'C:\\Clg\\TekWorks\\Datasets\\mercedes_benz_listings_cleaned.csv')\n",
    "df = pd.DataFrame(data)\n",
    "\n",
    "x = df[['Vehicle_Age']]\n",
    "y = df['Price_USD']\n",
    "\n",
    "model = LinearRegression()\n",
    "model.fit(x, y)\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"Mercedes-Benz Price Predictor\")\n",
    "st.write(\"Predict the price of a Mercedes-Benz based on Vehicle Age\")\n",
    "\n",
    "arrr = st.number_input(\"Enter Vehicle Age (in years):\", min_value=0, max_value=50, value=5)\n",
    "\n",
    "if st.button(\"Predict Price\"):\n",
    "    y_pred = model.predict(pd.DataFrame({'Vehicle_Age': [arrr]}))\n",
    "    st.success(f\"Predicted Price: ${y_pred[0]:,.2f}\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
