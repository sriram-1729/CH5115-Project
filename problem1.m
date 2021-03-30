% Problem 1
% Implementing the Bayesian Online Changepoint Detection algorithm

%% Initialization and loading data
clear;
load('NMRlogWell.mat');
N = length(y);
xx = 1:N;

%% Step 1: Setting intial conditions
lambda_CP = 250;
hazard = 1/lambda_CP;

mu_0 = 1.15;
k = 0.01;
alpha = 20;
beta = 2;

chi = zeros(4, 1);
chi(:, 1) = [mu_0; k; alpha; beta];

log_message(1) = 0;

log_R = -Inf(N+1, N+1);
log_R(1, 1) = 0;

mean_cap = zeros(1, N+1);
mean_cap(1) = mu_0;
lambda_cap = zeros(1, N+1);
lambda_cap(1) = k;

%% Starting the main loop
for i = 1:N
    % Step 2: Observing datum
    x = y(i);

    % Step 3: Calculating UPM predictive
    log_UPM_predictive = log(pdf('tLocationScale', x, chi(1, 1:i), ...
        sqrt((chi(4, 1:i) .* (chi(2, 1:i) + 1)) ./ ...
        (chi(3, 1:i) .* chi(2, 1:i))), 2 .* chi(3, 1:i)));
    
    % Step 4: Calculating growth probabilities
    log_growth_prob = log_UPM_predictive + log_message + log(1 - hazard);
    
    % Step 5: Calculating changepoint probabilities
    log_cp_prob = log(sum(exp(log_UPM_predictive) .* exp(log_message) ...
        * hazard));
        
    % Step 6: Computing joint and evidence, and normalizing joint
    log_new_joint = [log_cp_prob log_growth_prob];
    
    log_new_joint = log_new_joint - log(sum(exp(log_new_joint)));
    
    % Step 7: Compute RL posterior
    log_R(i+1, :) = [log_new_joint -Inf(1, N - i)];
    
    % Step 8: Update statistics and pass message
    chi = update_statistics(i, y, mu_0, beta, chi);
    
    log_message = log_new_joint;
    
    % Step 9: Perform prediction
    mean_cap(i+1) = sum(exp(log_R(i+1, 1:i+1)) .* chi(1, :));
    lambda_cap(i+1) = sum(exp(log_R(i+1, 1:i+1)) .* chi(2, :));
end

%% Plotting run length vs time
figure (1);
[~, run_length] = max(log_R, [], 2);
plot(run_length);
hold on;
title('Run length vs time');
xlabel('Time');
ylabel('Run length');
hold off;

%% Plotting given data and predictive mean vs time
figure (2);
plot(xx, y, xx, mean_cap(2:N+1));
hold on;
legend('Given data', 'Predictive mean');
title('Given data and predictor vs time');
xlabel('Time');
ylabel('Value');
hold off;





function z = update_statistics(i, y, mu_0, beta, chi)
    z = [chi(1, :) 0;
        chi(2, :) 0;
        chi(3, :) 0;
        zeros(1, length(chi(4, :)) + 1)];
    x = y(i);
    
    temp = (chi(1, :) .* chi(2, :) + x) ./ (chi(2, :) + 1);
    z(1, :) = [mu_0 temp];
    
    z(2, i+1) = z(2, i) + 1;
    
    z(3, i+1) = z(3, i) + 0.5;
    
    temp = chi(4, :) + (chi(2, :) .* (x - chi(1, :)) .^ 2) ...
        ./ (2 .* (chi(2, :) + 1));
    z(4, :) = [beta temp];
end