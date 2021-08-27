function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = Shekel(x, runlength, problemRng, seed)

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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% SHEKEL FUNCTION
%
% Authors: Sonja Surjanovic, Simon Fraser University
%          Derek Bingham, Simon Fraser University
% Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
%
% Copyright 2013. Derek Bingham, Simon Fraser University.
%
% For function details and reference information, see:
% http://www.sfu.ca/~ssurjano/
    

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
    d = 10;
    
    [a, b] = size(x);
    if (a == 1) && (b == d)
        theta = x'; %theta is a column vector
    elseif (a == d) && (b == 1)
        theta = x;
    else
        fprintf('x should be a column vector with %d rows\n', d);
        fn = NaN; FnVar = NaN; FnGrad = NaN; FnGradCov = NaN;
        return;
    end
    xx = theta'; % Convert to row vector
    
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
    CostGrad = zeros(runlength,d);
    hess=2^(-d+4);
    m = 10;
    fac=1;
    b = fac*0.1 * [1, 2, 2, 4, 4, 6, 3, 7, 5, 5]';
    C=[4*ones(d,1),1*ones(d,1),8*ones(d,1),6*ones(d,1),repmat([3,7]',d/2,1),...
        repmat([2,9]',d/2,1),repmat([5,3]',d/2,1),...
        repmat([8,1]',d/2,1),repmat([6,2]',d/2,1),repmat([7,3.6]',d/2,1)];
   % C = [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0;
    %     4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6;
     %    4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0;
      %   4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6];
    
    % Run simulation
    for i = 1:runlength
            
        % Start on a new substream
        if iscell(problemRng)
            DurationStream.Substream = seed + i - 1;
        end
        
        % Generate random noise data
        %Noise = normrnd(0, 1,[1,3]);
        
        outer = 0;
        for ii = 1:m
            bi = b(ii);
            inner = 0;
            for jj = 1:d
                xj = xx(jj);
                Cji = C(jj, ii);
                inner = inner + hess*(xj-Cji)^2;
            end
            outer = outer + 1/(inner+bi);
        end

        cost(i) = -outer+normrnd(0, 1);
        
        %g1 = 2*a*(x(2) - b * x(1)^2 + c * x(1) - r)*(-2* b * x(1) + c) - s*(1-t)*sin(x(1)) + Noise(2);  
        %g2 = 2*a*(x(2) - b * x(1)^2 + c * x(1) - r) + Noise(3);
        CostGrad(i, :) = zeros(1,d);
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
