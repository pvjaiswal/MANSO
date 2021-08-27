function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = Qtum(x, runlength, problemRng, seed)

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
        dim = 10;
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
    if iscell(problemRng)
        % Get random number stream from input and set as global stream
        DurationStream = problemRng{1};
        RandStream.setGlobalStream(DurationStream);
    end
        % Initialize for storage
        cost = zeros(runlength, 1);
        CostGrad = zeros(runlength, dim);
       
        %Ne=10000;
        % Run simulation
        for i = 1:runlength
            
            % Start on a new substream
            %DurationStream.Substream = seed + i - 1;

            % Generate random noise data
            path=pwd;
            %cd("/Users/prateekjaiswal/OneDrive - purdue.edu/stochastic_multistart/code/simopt_22/Problems/Qtum");
            if isfile('Qtum.m')
                cd(path);
            else
                path1 = fileparts(pwd);
                cd(path1+"/Problems/Qtum");
            end
            %setenv(pwd,'');
            %xtr=string(x);
            %pwd
            %[status,Output]=system("/Users/prateekjaiswal/anaconda3/bin/python "+"peterson_for_prateek_5.py -p "+xtr(1)+" "+xtr(2)+" "+xtr(3)+" "+xtr(4)+" "+xtr(5)+" "+...
            %    xtr(6)+" "+xtr(7)+" "+xtr(8)+" "+xtr(9)+" "+xtr(10));
            %cd('/Users/prateekjaiswal/OneDrive - purdue.edu/stochastic_multistart/code/simopt_22/Problems');
            dlmwrite('x_in', x, 'delimiter', ' ', 'precision', 16);
            %dlmwrite('x_in_done',1);
            while exist('y_out','file')==0 %&& exist('y_out_done','file')==0
                    pause(0.05); 
            end
%             while isempty(textscan(fopen("y_out"),'%f'))==1
%                 pause(0.01); 
%             end
            d=fopen("y_out");    
            co=textscan(d,'%f');
            %co=textread('y_out');
            %dlmread('y_out')
            %co{1}
            tees=0;
            if isempty(co{1})
                tees=1;
                while isempty(co{1})==1
                    d=fopen("y_out");    
                    co=textscan(d,'%f');
                    pause(0.05); 
                    fclose(d);
                end
            end
            cost(i)= double(co{1});%(co{1});%
            delete('y_out');
            %delete('y_out_done');
            if tees~=1
               fclose(d);
            end
            cd(path);
            %if status ==0
                %double(string(Output));
            %else
            %    fprintf('Simulator cannot be executed.\n');
            %end   
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
%matlab -nodisplay -nosplash -nodesktop -r "RunWrapper({'SAN'},{'ASTRDF'},1);exit;" | tail -n +11
%taskset -c <124> mpirun -np #  matlab -nodisplay -nosplash -nodesktop -r "RunWrapper({'SAN'},{'ASTRDF'},1);exit;" | tail -n +11 > log.txt 2>&1 &
