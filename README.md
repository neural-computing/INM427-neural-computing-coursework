Load all functions. 
Initial_Script.m (badly named) is the script that should be used. Grid Search Parameters can be set within it and these are passed onto the NN_model.m script.
This code uses the octave package "parallel" since octave does not natively have any compatability with parallel loops. This plus small synatic differences are likely to mean that the code will need modifying to run on pure Matlab.
