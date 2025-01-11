# Zyfra - Optimizaing Gold Recovery

## Site
[Zyfra - Gold Recovery Optimization](https://umbertofasci.github.io/Projects/OptimizingGoldRecovery.html)

## Introduction

In the mining industry, extracting gold from ore is a complex process that requires multiple stages of purification and refinement. The efficiency of this process, measured by the recovery rate, is crucial for both economic and environmental reasons. This project focuses on developing a machine learning model to predict the recovery rate of gold during the purification process, using data collected from various stages of a gold recovery plant.

The dataset encompasses multiple parameters measured throughout the technological process, including concentrations of different metals (Au, Ag, Pb), particle sizes, and various other features recorded at different stages of purification. These measurements are time-stamped, creating a temporal dimension to our analysis that could reveal important patterns in the recovery process.

The primary objective is to create a model that can accurately predict the recovery rate of gold, which will help optimize the purification process and reduce production costs.

The project's success will be measured using the Symmetric Mean Absolute Percentage Error (sMAPE), providing a balanced assessment of our model's predictive capabilities.


## Technological Process

In order to develop a proper workflow for optimization we must first understand what is to be optimized. In this case, understanding the technological process of gold extraction is essential.
Mined ore undergoes primary processing to get the ore mixture or rougher feed, which is the raw material for ploatation (also known as the rougher process). After flotation, the material is
sent to a two-stage purification.

```mermaid
flowchart TD
    A[Gold ore mixture] --> |input| B[/Flotation\]
    B --> |process| C(Rougher concentrate)
    B --> |tails| D@{ shape: procs, label: "Rougher tails" }
    C --> E(First stage of cleaner process)
    E --> |tails| D
    E --> F(Second stage of cleaner process)
    F --> |tails| D
    F --> |output| G[Final concentrate]
```

### Flotation
- Gold ore mixture is fed into the float banks to obtain rougher Au concentrate and roughter `tails` (product residues with low concentration of valuable metals).
- The stability of this process is affected by the volatile and non-optimal physicochemical state of the `flotation pulp` (a mixture of solid particles and liquid). 

### Purification
- The rougher concentrate undergoes two stages of purification. After purification, the final concentrate and new tails is obtained.

### Staging & Process Components

Below are the staging and processing components associated with the technological process, these compnents are utilized as features in the final dataset.

```mermaid
graph LR
    subgraph Process Components
        direction TB
        RF[Rougher feed]
        
        subgraph Reagents[Rougher/Reagent additions]
            direction LR
            X[Xanthate<br/>Promoter/activator]
            S[Sulphate<br/>Sodium sulphide]
            D[Depressant<br/>Sodium silicate]
        end
        
        RP[Rougher process<br/>Flotation]
        RT[Rougher tails<br/>Product residues]
        FB[Float banks<br/>Flotation unit]
        CP[Cleaner process<br/>Purification]
        RAu[Rougher Au<br/>Rougher gold concentrate]
        FAu[Final Au<br/>Final gold concentrate]
    end
    
    subgraph Stage Parameters
        direction TB
        AA[Volume of air]
        FL[Fluid levels]
        FS[Feed particle size]
        FR[Feed rate]
    end
```
## Data Description

Given the technological process which the features originate frome they are named in the following format:

```
[stage].[parameter_type].[parameter_name]
```
With this in mind we can take another look at the technological process where we can then infer the feature
names present in the dataset.

```mermaid
flowchart TD
    A[Gold ore mixture] --> |1| B[/Flotation\]
    B --> |3| C(Rougher concentrate)
    B --> |2| D@{ shape: procs, label: "Rougher tails" }
    C --> E(First stage of cleaner process)
    E --> |4| D
    E --> |5| F(Second stage of cleaner process)
    F --> |6| D
    F --> |7| G[Final concentrate]
```

<div align="center">
    
| # | Feature Name |
|--------|------------|
| 1 | rougher.input.feed_au |
| 2 | rougher.output.tail_au |
| 3 | rougher.output.concentrate_au |
| 4 | primary_cleaner.output.tail_au |
| 5 | primary_cleaner.output.concentrate_au |
| 6 | secondary_cleaner.output.tail_au |
| 7 | final.tail_au |

</div>

## Recovery Calculation Function

In the procedure, the recovery calculation of the provided dataset is tested against one developed by myself. This is done after importing
and ingesting the data which came in three separate files: `gold_recovery_train.csv`, `gold_recovery_test.csv`, and `gold_recovery_full.csv`.

>[!NOTE]
> The downloaded data is straight from the source, and some parameters are not available becasue they were measured or calculated
> after the data was collected. That's why some features that are present in the training set may be absent from the test set. 

Now, continuing with the task at hand. The first objective with this dataset was to make sure that the recovery was calculated appropriately.
To do this, the following equation was used:

```math
Recovery = \frac{C \times (F - T)}{F \times (C - T)} \cdot 100\%
```

Where,


- $C$ — share of gold in the concentrate right after flotation (for finding the rougher concentrate recovery)/after purification (for finding the final concentrate recovery)
- $F$ — share of gold in the feed before flotation (for finding the rougher concentrate recovery)/in the concentrate right after flotation (for finding the final concentrate recovery)
- $T$ — share of gold in the rougher tails right after flotation (for finding the rougher concentrate recovery)/after purification (for finding the final concentrate recovery)

Calculating the recovery in python can be achieved with:
```python
def calculate_recovery(row, concentration_col, feed_col, tails_col):
    C = row[concentration_col]
    F = row[feed_col]
    T = row[tails_col]
    
    # Avoid division by zero
    if F == 0 or (C - T) == 0:
        return 0
    
    recovery = C * (F - T) / (F * (C - T)) * 100
    
    # Handle edge cases
    if np.isnan(recovery) or np.isinf(recovery):
        return 0
    
    return recovery
```

Applying this function to the training set:
```python
train_df['calculated_recovery'] = train_df.apply(
    lambda row: calculate_recovery(
        row,
        'rougher.output.concentrate_au',  # C
        'rougher.input.feed_au',          # F
        'rougher.output.tail_au'          # T
    ),
    axis=1
)

```
Validating the recovery calculation:
```python
valid_recovery_mask = (~train_df['rougher.output.recovery'].isna()) & (~train_df['calculated_recovery'].isna())

mae = mean_absolute_error(
    train_df.loc[valid_recovery_mask, 'rougher.output.recovery'],
    train_df.loc[valid_recovery_mask, 'calculated_recovery']

)
```
>[!NOTE]
> The `valid_recovery_mask` is created as a way to only include values that are not missing and to avoid any errors.
> Keep in mind this is being done before any initial data processing. 

### Recovery Calculation Results

The Mean Absolute Error between the calculated and the actual recovery values was found to be **`9.30e-15`**.
Considering this extremely low MAE value, effectively zero, we confirmed that the implementation of the recovery formula
matches the existing calculations in the dataset.

This near-zero difference validates both our understanding of the recovery calculation process and the reliability of
the provided data, ensuring a solid foundation for subsequent modeling efforts.

## Raw Data Dilemmas

### Missing Test Features

To better describe what features are missing in the provided test set here is a complete list of the missing 35 features.

<details>
<summary>Click to view all missing features</summary>
    
- final.output.concentrate_ag (Type: float64)
- primary_cleaner.output.tail_pb (Type: float64)
- rougher.calculation.floatbank11_sulfate_to_au_feed (Type: float64)
- rougher.calculation.floatbank10_sulfate_to_au_feed (Type: float64)
- rougher.output.tail_sol (Type: float64)
- rougher.output.concentrate_sol (Type: float64)
- final.output.tail_ag (Type: float64)
- primary_cleaner.output.concentrate_ag (Type: float64)
- rougher.calculation.sulfate_to_au_concentrate (Type: float64)
- rougher.output.tail_pb (Type: float64)
- primary_cleaner.output.tail_ag (Type: float64)
- calculated_recovery (Type: float64)
- rougher.calculation.au_pb_ratio (Type: float64)
- secondary_cleaner.output.tail_sol (Type: float64)
- secondary_cleaner.output.tail_au (Type: float64)
- secondary_cleaner.output.tail_pb (Type: float64)
- primary_cleaner.output.concentrate_sol (Type: float64)
- final.output.concentrate_sol (Type: float64)
- primary_cleaner.output.tail_au (Type: float64)
- final.output.tail_sol (Type: float64)
- rougher.output.recovery (Type: float64)
- primary_cleaner.output.concentrate_au (Type: float64)
- rougher.output.concentrate_au (Type: float64)
- rougher.output.tail_ag (Type: float64)
- final.output.concentrate_pb (Type: float64)
- final.output.concentrate_au (Type: float64)
- final.output.tail_au (Type: float64)
- rougher.output.concentrate_ag (Type: float64)
- primary_cleaner.output.concentrate_pb (Type: float64)
- secondary_cleaner.output.tail_ag (Type: float64)
- rougher.output.concentrate_pb (Type: float64)
- rougher.output.tail_au (Type: float64)
- final.output.recovery (Type: float64)
- final.output.tail_pb (Type: float64)
- primary_cleaner.output.tail_sol (Type: float64)

</details>

An easy way to obtain this list of missing features in the test set that are present in the training is to utilize the `set()` functionality:

```python
train_columns = set(train_df.columns)
test_columns = set(test_df.columns)

missing_in_test = train_columns - test_columns
```

### Missing Values

Here we highlight the inherent challenges in industrial process data collection. The training set exhibits 85 columns with missing values, with significant gaps in final output measurements
and recovery data, notably 1,521 missing values in final output recovery. State measurements such as floatbank levels and air parameters show consistent patterns of 85 - 103 mising entries.
The test set demonstrates a more extensive pattern of missing values, particularly in input and state parameters. While floatbank state measurements chonsistently express around 16 missing
values, chemical input measurements display higher variability in missing data, randing from 166 to 353 missing values for xanthate measurements and up to 302 for sulfate readings. Feed
measurements also show varying degrees of missingness across different parameters.

### Missing Data Imputation

#### Bi-Directional Rolling Average Interpolation

In this section an experiment in filling time-series-like data is performed targeting several methodologies and data leakage. While dropping the values was
an option, at this point I wanted to try and maintain the temporal effects of the dataset. The first approach that was made was to conduct a **rolling average interpolation**.
This interpolation formula creates a bidirectional moving average by computing both forward and backward rolling means with a specified window size, then averages these two
directions to produce more balanced estimates.

```math
Combined  Average(t) = \frac{1}{2} (\frac{\sum_{i=t}^{t+w} X_{i}}{w} + \frac{\sum_{i=t-w}^{t} X_{i}}{w})
```
where,

- $w$ — window size
- $t$ — current time point
- $X_{i}$ — value at time i

This is performed in python:

```python
def rolling_average_interpolate(series, window):
    # Create forward and backward rolling means
    forward_roll = series.rolling(window=window, min_periods=1).mean()
    backward_roll = series[::-1].rolling(window=window, min_periods=1).mean()[::-1]
    
    # Combine forward and backward rolls
    combined_roll = (forward_roll + backward_roll) / 2
    
    # Only fill the NaN values in the original series
    result = series.copy()
    result[series.isna()] = combined_roll[series.isna()]
    return result
```
