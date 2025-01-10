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
    A[Gold ore mixture] --> |input| B(Flotation)
    B --> |process| C(Rougher concentrate)
    B --> |tails| D[Rougher tails]
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
    A[Gold ore mixture] --> |1| B(Flotation)
    B --> |3| C(Rougher concentrate)
    B --> |2| D[Rougher tails]
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

