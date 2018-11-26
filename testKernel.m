X = gpuArray(single(input));
Y = gpuArray(single(target));
wX = X;
[N, P] = size(X);
lambda = [0.123, 0.0123];
nLambda = 2;
lambdaMax = 1.;
alpha = 0.5;
maxIter = 100;
standardize = true;
muX = 0;
sigmaX = 1;
B = lassoGPUKernel(X0, wX0, Y0, N, P, lambda, nLambda, lambdaMax, maxIter, alpha, standardize , muX, sigmaX);


