function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = Branin(x, runlength, problemRng, seed)

% INPUTS
% x: a column vector equaling the decision variables theta
% runlength: the number of longest paths to simulate
% problemRng: a cell array of RNG streams for the simulation model 
% seed: the index of the first substream to use (integer >= 1)

% RETURNS
% Estimated fn value
% Estimate of fn variance
% Estimated gradient. This is an IPA estimate so is the TRUE gradient of
% the estimated function value
% Estimated gradient covariance matrix
%%%%%%%%%%
% Branin-Hoo function is defined on the square x1 ∈ [-5, 10], x2 ∈ [0, 15].
% 
%  It has three minima with f(x*) = 0.397887 at x* = (-pi, 12.275),
%     (+pi, 2.275), and (9.42478, 2.475).
% 
%     More details: <http://www.sfu.ca/~ssurjano/branin.html>


constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;

if (runlength <= 0) || (round(runlength) ~= runlength) || (seed <= 0) || (round(seed) ~= seed)
    fprintf('runlength should be a positive integer,\nseed should be a positive integer\n');
    fn = NaN;
    FnVar = NaN;
    FnGrad = NaN;
    FnGradCov = NaN;
    
else % main simulation
    %numarcs = 13;
    %numnodes = 9;
    dim = 2;
    [a, b] = size(x);
    if (a == 1) && (b == dim)
        theta = x'; %theta is a column vector
    elseif (a == dim) && (b == 1)
        theta = x;
    else
        fprintf('x should be a column vector with %d rows\n', dim);
        fn = NaN; FnVar = NaN; FnGrad = NaN; FnGradCov = NaN;
        return;
    end
    x = theta'; % Convert to row vector
    
    % Get random number stream from input and set as global stream
    if iscell(problemRng)
%         problemRng = cell(1, NumRngs);
%                 for i = 1:NumRngs
%                     problemRng{i} = RandStream.create('mrg32k3a', 'NumStreams', (2 + NumRngs)*repsAlg, 'StreamIndices', (2 + NumRngs)*(j - 1) + 2 + i);
%                 end
        DurationStream = problemRng{1};
        RandStream.setGlobalStream(DurationStream);
    end
    
    % Initialize for storage
    cost = zeros(runlength, 1);
    CostGrad = zeros(runlength, dim);
    
    a=1;
    b=5.1 / (4 * pi^2);
    c=5.0/ pi;
    r=6;
    s=10;
    t=1.0 / (8 * pi); 
    
    % Run simulation
    for i = 1:runlength
            
        % Start on a new substream
        if iscell(problemRng)
            DurationStream.Substream = seed + i - 1;
        end
        
        % Generate random noise data
        Noise = normrnd(0, 1,[1,3]);
        
        
        cost(i) = (a * (x(2) - b * x(1)^ 2 + c * x(1) - r)^ 2 + s * (1 - t) * cos(x(1)) + s) + Noise(1);
            
        g1 = 2*a*(x(2) - b * x(1)^2 + c * x(1) - r)*(-2* b * x(1) + c) - s*(1-t)*sin(x(1)) + Noise(2);  
        g2 = 2*a*(x(2) - b * x(1)^2 + c * x(1) - r) + Noise(3);
        CostGrad(i, :) = [ g1 , g2 ];
    end
    
    % Calculate summary measures
    if runlength==1
        fn=cost;
        FnVar=0;
        FnGrad=CostGrad;
        FnGradCov=zeros(length(CostGrad));
    else
        fn = mean(cost);
        FnVar = var(cost)/runlength;
        FnGrad = mean(CostGrad, 1); % Calculates the mean of each column as desired
        FnGradCov = cov(CostGrad); %FnGradCov = cov(CostGrad, 2);
    end
end
%RunWrapper({'Branin'},{'ASTRDF'},10)
%PostWrapper({'Branin'},{'ASTRDF'},50)

%I  think most of the comments are resolved now. I will  update the paper with result soon and let you know.