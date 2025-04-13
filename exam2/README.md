# Exam 2 Portion 2

## 2. Dataset Assessment

## 3. Variables

Facility Name and ID are associated with the respective facility. Given possibility of having a duplicate Facility Name, the ID serves as a unique identifier. State is where in the country the facility is located. Measure Name is a specific metric/condition that the readmission data is being reported for. Number of Discharges is the number associated with patients being released from the hospital, this is for patients involved in the Measure Name. Footnote is for any additional remarks, in this dataset, it appears to be primarily integers, might be additional patients that were discharged after the total had been calculated. Excess Readmission Ratio is the ratio of actual number of readmissions to number expected. Predicted Readmission Rate is the predicted number of patients readmitted based on the facility's history. Expected Readmission Rate is the expected rate based on national average. Number of Readmissions is the actual number of patients that get readmitted for the Measure Name. Start and End Date are the starting and ending period for the data collection.

### Data Types

Facility Name String
Facility ID Integer
State Char
Measure Name String
Number of Discharges Float
Footnote Float
Excess Readmission Ratio Float
Predicted Readmission Rate Float
Expected Readmission Rate Float
Number of Readmissions Integer
Start Date DateTime Obj
End Date DateTime Obj

## 4. Research Question

What factors within a facility lead to readmission rates being higher for a particular measure over another?

## 5. Why this Question?

Some illnesses, medical conditions, etc. may require more treatment and attention than say a simple cold might. If the dataset is defining readmission as an unplanned return with a 30 day period, what factors within the facility are contributing to that readmission? Is is something to do with medical care, lack of treatment, medication issues, sudden complications leading to a follow up? Is the number of readmissions skewed based on the medical condition ("Measure Name") too? What if within the vicinity we had a large population that caught a really bad flu and they were all in and out and in again within a 30 day period; whereas we had a small group that had a particular disease and they too had to be readmitted. I'm curious to know if we can use "Measure Name" to identify if there are any possible trends with "Number of Readmissions", "Facility Name/ID", and "State".

## 6. Chosen Algorithm

Random Forest as my algorithm. Decided to use Random Forest as there were a lot of predictors that could affect the target feature. I wanted to see how they would affect it and what the outcome would be.

## 7. Algorithm Validation

K-Fold Cross Validation.

## 8.

### Disclosures

Used a Gen AI for the Random Forests Model and debugging.

Additional references:
https://medium.com/@enozeren/building-a-random-forest-model-from-scratch-81583cbaa7a9
https://github.com/enesozeren/machine_learning_from_scratch/blob/main/decision_trees/random_forest.py
https://github.com/enesozeren/machine_learning_from_scratch/blob/main/decision_trees/decision_tree.py

### How to Run

Within the folder should be a file named main.py. Just open it and run, everything else should follow.

## 9. Summary
