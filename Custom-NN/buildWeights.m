function w = buildWeights(i_size, j_size, epsilon)
  w = rand(j_size, i_size) * 2 * epsilon - epsilon;
  % creates a matrix of j_size rows and i_size columns within +- epsilon
end