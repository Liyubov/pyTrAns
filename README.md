# pyTrAns package
PyTrAns (Trajectories Analysis) is python programs for trajectories analysis. The package has several modules, which one can use independently from each other for
1. generating or loading trajectories `loading_trajectories.py`
2. statistical analysis of trajectories (including scaling exponent estimation) `convex_hull.py`
3. classification and post-analysis of trajectories based on analysis modules

There are several types of trajectories we generate and analyse:

    from random generating processes (generated by stochastic system of equations)
    from real trajectories (from observations) see description and credits in generating_trajectories file

# Data 
1. Data for testing package can be generated (see generating or loading_trajectories files). 
2. Data examples can be found on website in challanges such as https://competitions.codalab.org/ and https://github.com/AnDiChallenge/ANDI_datasets

# Usage 
Please follow the notebooks to see how all functions are working. 
The main functions are: generating_trajectories.py, loadng_trajectories and modules for calculation of distributions etc. 
We recommend you to read more about stochastic processes analysis in the papers e.g. here https://sites.google.com/view/fellowshipresultsliubov/research-projects/random-walks-analysis-and-applications?authuser=0

The package is under development, for using it you need to use `import PROGRAM_NAME` e.g. `import convex_hull_analysis as *`


## Theoretical analysis 
Here we propose investigation of properties of random walks, such as gyration radius, msd, number of sites visited by random walk and some other properties.

(work in progress)

# Types of Contribution

There are multiple ways to contribute to netrd (borrowed description of contribution from netrd).

## Report Bugs

To report a bug in the package, open an issue at https://github.com/Liyubov/pyTrAns/issues

Please include in your bug report:

    Your operating system name and version.
    Any details about your local setup that might be helpful in troubleshooting.
    Detailed steps to reproduce the bug.

## Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.
Implement Features or New Methods

# Credits
While preparing this software some other open packages were used, which are mentioned in notebooks and code (with the MIT license).
This is work in progress.
