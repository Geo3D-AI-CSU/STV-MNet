Program Description:

DL___AutomaticTreeInventory.r is the program that does not save the Wr results.
Output: /results/Structure calculation/results0.1/

Modify___AutomaticTreeInventory.r is the program that saves the corresponding Wr results.
Output: /results/Structure calculation/results_Wr

appendPixelW.py is the program that connects the structure calculation results with Wr.

After merging all CSV files:

Calculate the ratio of Wr to 0.03.
Multiply the structural parameters calculated before modifying Wr by this ratio to obtain the corrected structural parameters.
