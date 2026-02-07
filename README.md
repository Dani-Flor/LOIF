# Power System Line Outage Experimental Platform

This experimental platform, composed of MATLAB and Python code, consists of four components to conduct experiments to evaluate the performance of feature selection techniques for power system line outage detection using power measurements with a classification algorithm (e.g., k-nearest neighbors).

## Component 1 (MATLAB): Labeled Dataset Generator
The first component of the platform, consisting of the source files: 1. _lod_labeled_datagen.m_ and 2. _LOIF_matrix.m_, runs a power system simulation using the MATPOWER library, generates all possible line outage conditions, and collects all power measurements for each of those conditions. A dataset with the power measurements labeled with the line outage condition is created as a CSV file.

## Component 2 (Python): Detectable Subset Generator
The second component of the platform, inside the source file: _lod_experiments.py_, determines for each line in a power system, the set of lines whose outages could be detected by observing the power measurements on this line. The Line Outage Distribution Factor (LODF) and Line Outage Impact Factor (LOIF) are used for this determination.

## Component 3 (Python): Observed Transmission Line Selector
The third component of the platform, inside the source file: _lod_experiments.py_, selects the best observed transmission lines (i.e., those whose power measurements should be used as features in the line outage detection classification problem).

## Component 4 (Python): Classification Experiment Execution
The fourth and final component of the platform, inside the source file: _lod_experiments.py_, conducts classification experiments to benchmark the classification performance of the features selected for the line outage detection classification problem.

**More information on this platform can be found in the extended documentation: _lod_exp_platform_readme.docx_**
