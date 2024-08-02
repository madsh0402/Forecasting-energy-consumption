---
---

```python
from IPython.display import display, HTML

import pandas as pd
import numpy as np

import matplotlib as plt
import seaborn as sns
```

## 4.3 Overview of the explanatory variables

To make a good forecast model for energy consumption, it is important to understand the attributes that the variables have and know what each variable represents. Each variable serves to give a better understanding of what contributes to energy consumption. By understanding the variables, a better understanding of energy usage patterns can be gained. Therefore, the following section will introduce and describe the variables contained within the dataset.

| Variable Name | Data Type | Variable Type | Description | Source | Additional Notes |
|---------------|-----------|---------------|-------------|--------|------------------|
| start_time | datetime | Numerical | Start time of the observation | Gathered from energidataservice.dk  | - |
| is_holiday | int | Binary | Indicates if the time is a holiday | python library *holidays* | 1 for holiday, 0 for non-holiday |
| month | str | Categorical | Month of the observation | python library *calender* | - |
| weekday | str | Categorical | Weekday of the observation | python library *calender* | - |
| EC_pct_change | float | Numerical | Percent change in electrical cars registrated | Gathered from DST.dk | - |
| HC_pct_change | float | Numerical | Percent change in plug-in hybrid cars registrated | Gathered from DST.dk | - |
| humidity_past1h | float | Numerical | Humidity level in the past hour | Gathered from DMI.dk | - |
| temp_mean_past1h | float | Numerical | Mean temperature in the past hour | Gathered from DMI.dk | - |
| wind_speed_past1h | float | Numerical | Wind speed in the past hour | Gathered from DMI.dk | - |
| EL_price | float | Numerical | Price of electricity | Gathered from DST.dk | - |
| GrossConsumptionMWh | float | Numerical | Sum of energy consumption | Gathered from energidataservice.dk | Dependent variable for analysis |

- **start_time**: Timestamps marking the beginning and end of the recorded hour.
- **is_holiday**: A binary variable indicating whether the observation falls on a holiday.
- **month & weekday**: Time-related categorical variables aiding in temporal pattern identification.
- **EC_pct_change & HC_pct_change**: Percentage changes in energy consumption and heating, respectively, offering insights into variability and trends.
- **umidity_past1h, temp_mean_past1h, wind_speed_past1h**: Meteorological variables from the previous hour that might influence energy usage patterns.
- **EL_price**: Price of electricity, a potential determinant of consumption behavior.
- **GrossConsumptionMWh**: Energy consumption, serving as the dependent variable for analysis.

The foundation of any data analysis lies in understanding the basic statistical properties of a dataset. Summary statistics provide essential information that can guide initial conclusions and is a way to get directions for future analysis. Therefore, the dataset’s summary statistics are provided to offer an overview of the variables, each of which plays a role in understanding the energy consumption patterns. The summary statistics of numerical variables are as follows:

## 4.4 Summary statistics insights of numerical variables


```python
df = pd.read_csv("C:/Users/madsh/OneDrive/Dokumenter/kandidat/Fællesmappe/Forecasting-energy-consumption/Data Cleaning/output_file.csv", encoding="utf-8")
```

```python
# Selecting the columns of interest for the summary statistics
columns_of_interest = ['Electric cars', 'Plug-in hybrid cars', 'humidity_past1h', 'temp_mean_past1h', 'wind_speed_past1h', 'EL_price','GrossConsumptionMWh']

# Calculating the summary statistics
summary_stats = df[columns_of_interest].describe().transpose()
# Adding the sum and count manually
summary_stats['count'] = df[columns_of_interest].count()

# Renaming columns to match your required format
summary_stats.columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']

# Rearranging the order of columns to match your required format
summary_stats = summary_stats[['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']].transpose()
summary_stats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Electric cars</th>
      <th>Plug-in hybrid cars</th>
      <th>humidity_past1h</th>
      <th>temp_mean_past1h</th>
      <th>wind_speed_past1h</th>
      <th>EL_price</th>
      <th>GrossConsumptionMWh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Count</th>
      <td>26304.000000</td>
      <td>26304.000000</td>
      <td>26304.000000</td>
      <td>26304.000000</td>
      <td>26304.000000</td>
      <td>26304.000000</td>
      <td>26304.000000</td>
    </tr>
    <tr>
      <th>Mean</th>
      <td>0.007992</td>
      <td>0.008540</td>
      <td>81.429244</td>
      <td>8.331720</td>
      <td>4.834021</td>
      <td>0.816396</td>
      <td>4077.927643</td>
    </tr>
    <tr>
      <th>Std</th>
      <td>0.004053</td>
      <td>0.005619</td>
      <td>10.658195</td>
      <td>6.150023</td>
      <td>1.990381</td>
      <td>0.904135</td>
      <td>771.035124</td>
    </tr>
    <tr>
      <th>Min</th>
      <td>-0.012073</td>
      <td>-0.027599</td>
      <td>39.800000</td>
      <td>-9.755000</td>
      <td>0.980000</td>
      <td>-0.365245</td>
      <td>2396.632690</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.005086</td>
      <td>0.003659</td>
      <td>75.193548</td>
      <td>3.242623</td>
      <td>3.319643</td>
      <td>0.219713</td>
      <td>3499.425812</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.007145</td>
      <td>0.008175</td>
      <td>84.360656</td>
      <td>8.131967</td>
      <td>4.551724</td>
      <td>0.465135</td>
      <td>4068.527038</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.010764</td>
      <td>0.010722</td>
      <td>89.606557</td>
      <td>13.295020</td>
      <td>6.061507</td>
      <td>1.107715</td>
      <td>4629.730163</td>
    </tr>
    <tr>
      <th>Max</th>
      <td>0.036147</td>
      <td>0.062953</td>
      <td>97.934426</td>
      <td>26.935593</td>
      <td>16.248214</td>
      <td>6.478240</td>
      <td>6664.007813</td>
    </tr>
  </tbody>
</table>
</div>



Looking at the growth patterns in the table above, the mean percentages for Electric cars and Plug-in hybrid cars indicate a steady growth trend, with means of 0.007992 and 0.008540, respectively. This steady growth, combined with the low standard deviations (0.004053 for Electric cars and 0.005619 for Plug-in hybrid cars), suggests a consistent, albeit modest, increase in new electric and hybrid vehicle registrations. The growth pattern shows a relatively narrow range between the maximum and minimum values: Electric cars range from -0.012073 to 0.036147, while Plug-in hybrid cars range from -0.027599 to 0.062953. This indicates that the market is expanding at a steady rate during the period of interest, without sudden spikes or drops. The difference in growth rates between electric and hybrid vehicles may reflect varied consumer preferences or policy incentives targeted at specific vehicle types. The slightly higher growth rate for hybrid vehicles might indicate a transitional preference among consumers, moving from conventional gas engines to fully electric options as technology and infrastructure improve.

However, looking at the period shortly after our scope shows the import of electric cars as a percentage change growth rate would be larger than that of the hybrid cars (Elbiler Udgjorde 36 Pct. Af De Nye Biler I 2023, n.d.). The environmental factors (humidity_past1h, temp_mean_past1h, and wind_speed_past1h) show a high degree of stability, with relatively low standard deviations (10.658195 for humidity_past1h, 6.150023 for temp_mean_past1h, and 1.990381 for wind_speed_past1h). This suggests that the observed period did not have significant environmental fluctuations that could have a large effect on energy consumption patterns. The direct impact of these environmental conditions on energy consumption, especially the effect they have on the need for heating or cooling, could be more nuanced. For instance, moderate temperatures might reduce the need for extensive heating or cooling, potentially moderating energy consumption related to climate control for private consumers.

Lastly, the electricity dynamics, consisting of EL_price and energy consumption (Sum_quantity), show minimal fluctuation in electricity price (standard deviation of 0.904135) compared to energy consumption (7.947110). This suggests that while electricity pricing might be regulated or have less volatility due to different means of production, the consumption patterns vary significantly among consumers over time. This variability could reflect different aspects that influence consumer behavior, not directly tied to price changes, such as seasonality and temporal patterns. The dataset includes private customers, who likely have clear usage patterns, using minimal electricity while at work and more when at home. The substantial difference between the 75th percentile (1.107715) and the maximum value (6.478240) for Sum_quantity points to sporadic high consumption events, possibly due to holidays like Christmas when consumers use more electricity-demanding systems simultaneously (Juleaften: Øerne Bruger Mest Ekstra Strøm, Mens Storbykommuner Bruger Mindst, 2022). This is accounted for with the IsHoliday variable, which helps capture these outliers.

The variable "GrossConsumptionMWh" represents the total energy consumption measured in megawatt-hours. This metric provides an overview of the overall energy demand within the observed period. The mean value of 4077.927643 MWh, with a standard deviation of 771.035124, indicates that there is considerable variation in energy consumption. The minimum and maximum values of 2396.632690 MWh and 6664.007813 MWh, respectively, further highlight this variability. The 25th percentile (3499.425812 MWh) and 75th percentile (4629.730163 MWh) values show that most consumption values lie within this range, suggesting that while there are periods of high consumption, the majority of the data points fall within a more consistent range. This variability in GrossConsumptionMWh can reflect differing energy needs across various times, influenced by factors such as seasonal changes, consumer behavior, and possibly the number of electric and hybrid cars being charged. Understanding this variable is crucial for energy providers and policymakers to ensure a stable supply and to implement strategies for efficient energy distribution and consumption.

Furthermore, the table highlights the different scales of the numerical variables, an aspect that can disrupt machine learning models' prediction accuracy. Therefore, it is evident from the tables that the numerical values should be scaled similarly before applying machine learning models. Analyzing the summary statistics allows a deeper exploration of the dataset's essential aspects: interrelations among variables, aggregating data over customer groups, and time-based dimensions. The next chapter is dedicated to examining these correlations, providing insight into the impact of these dimensions.
