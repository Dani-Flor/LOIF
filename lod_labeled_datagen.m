% function lod_labeled_datagen(mpc_string,ML,hourly_load_variation,percent_load_var,samples_per_scenario,dataset_label,rng_seed,DC_or_AC)
% system: Name of the case file in matpower that we are interested in (Ex. 'case118' -> 118 bus system)
% otl: This is the monitored line(s) which can be a single integer, an array of integers, or the sentinel value '[]' (all lines).
% hourly_var: Hourly Load Variation will be used later when we try to implement a daily load profile.
% rand_var: Percentage that represents load variation (random number from -% to +%).
% samples: Number of Samples For Each Scenario (Outages and Normal Conditions)
% label: Used to identify file name of Training Data
% rng_seed: The seed of Random Number Generator (used for random load variation).
% If we set rng_seed to 0, seed is Generated using POSIX time (number of seconds since 1970)
% If we want to replicate exact load variation, we enter the RNG seed of previous training data.
% sol_type: This is to determine if the user wants to collect results by running AC or DC solutions.
% 
% What is Happening?
% We Collect Labeled Training Data For The Following Scenarios
    % All single-line outages
    % System Under Normal Conditions
% We Collect Training Data for all scenarios when
    % The system has no load variation (base load)
    % When the system has a random load variation
    % Repeated Based on the number of samples
% 
% When disconnecting lines, generation is fixed based on the Optimal Power Flow (OPF) solution of the system under normal conditions.
% When simulating a line outage, we use regular Power Flow (PF) Solution.

function lod_labeled_datagen(system,otl,hourly_var,rand_var,samples,label,rng_seed,sol_type)
%% Calculate LOIF and LODF Matrices
LOIF_matrix(system)
%% Initalialization
tic;                              %Start clock
define_constants;                 % MATPOWER indices (PD,QD,BR_STATUS,PF,...)
warning('off','all');          
mpopt = mpoption('verbose',0,'out.all',0); %Disables MATPOWER output messages   
%load system
mpc_copy = loadcase(system); %Copy of the System before any changes are made
num_lines  = size(mpc_copy.branch,1); %Calculate the number of lines

%% Checks if user parameters are entered correctly
%Checks if sol_type is AC or DC, if not error
if (strcmp(sol_type,'AC')) | (strcmp(sol_type,'DC'))
else
    error('User Must Type "DC" or "AC"')
end

%checks if the otls entered are valid (exist in system), error if not.
%also checks if otls is equal to sentinal value (empty: []), if so consider
%all lines  
if isempty(otl)   % sentinel => all lines
    otl = 1:num_lines;
elseif ~all(ismember(otl,1:num_lines))
    error('System only has %d lines. ML contains out-of-range entries.', num_lines);
end

%RNG Seed
%If rng_seed = 0, Generate seed using posixtime
if rng_seed == 0
    seed = posixtime(datetime('now'));
else
    %Else, use seed provided by user
    seed = rng_seed;
end
rng(seed);

%% Initialize Variables
num_samples = samples + 1;
num_rows  = num_lines + 1;       
num_col  = numel(otl)*4 + 1;              % 4 features per line([PF,QF,PT,QF]) + label

labeled_dataset = zeros(num_samples*num_rows, num_col);
row = 0;  %used as counter

% convergence data
convergence_data = zeros(num_rows, num_samples+1);
convergence_data(:,1) = 0:num_lines;

previous_results = ones(num_rows,1);         %Convergence Results of All outages and normal condition of prev. sample

%Orignal PD/QD
PD_copy = mpc_copy.bus(:, PD);
QD_copy = mpc_copy.bus(:, QD);

for i = 0:samples
    
    col = i + 1;
    % Random load variation for this sample
    if i == 0
        Load_Variation = 0;
    else
        percent_var = rand_var * 0.01;
        a = -percent_var;
        b = percent_var;
        Load_Variation = (b-a) * rand() + a;  %This gives a random load variation between -percent: +percent
    end

    v = hourly_var * (1 + Load_Variation);
    %update load with variation
    PDs   = PD_copy * v;
    QDs   = QD_copy * v;
    mpc_updated = mpc_copy;  %Copy original system (No changes)
    mpc_updated.bus(:, PD) = PDs;    %Update Active Demand with Load Variation
    mpc_updated.bus(:, QD) = QDs;    %Update Reactive Demand with Load Variation

    % Convergence results for current sample
    current_results = zeros(num_rows,1);
    %% For Normal Conditions
    if previous_results(1) == 1  %If previous sample converged for normal conditions
        mpc_norm = mpc_updated;

        if strcmp(sol_type,'AC')
            results0 = runopf(mpc_norm, mpopt);  %AC OPF
        else
            results0 = rundcopf(mpc_norm, mpopt); %DC OPF
        end

        if results0.success == 1
            gen_dat = results0.gen;  %Collect Gen Data(Normal conditions) to fix generation for outage conditions
            current_results(1) = 1;

            % Collect features
            f = results0.branch(otl, [PF QF PT QT]); 
            Dfeat = reshape(f.', 1, []);
     
            row = row + 1;
            labeled_dataset(row,:) = [Dfeat, 0];
        else
            % If x=0 fails, mark and skip all outages for this sample
            previous_results(1) = 0;
            current_results(1) = 0;
            % no gen_dat available -> skip remaining x
            convergence_data(:, col+1) = current_results;
            continue;
        end
    else
        % previously failed, always skip
        current_results(1) = 0;
        convergence_data(:, col+1) = current_results; %save results for in convergence matrix
        continue;
    end

    %% For All Possible Outages
    for x = 1:num_lines
        if previous_results(x+1) == 0
            % previously failed => skip forever
            current_results(x+1) = 0;
            continue;
        end

        mpc_out = mpc_updated;
        mpc_out.branch(x, BR_STATUS) = 0;     % disconnect line x
        mpc_out.gen = gen_dat;               % fixed generation from normal OPF

        if strcmp(sol_type,'AC')
            results = runpf(mpc_out, mpopt);  %AC power flow solution
        else
            results = rundcpf(mpc_out, mpopt);  %DC power flow solution
        end

        if results.success ~= 1
            % mark as failed
            previous_results(x+1) = 0;
            current_results(x+1) = 0;
            continue;  %skip feature collection
        end

        current_results(x+1) = 1;

        % Collect features
        f = results.branch(otl, [PF QF PT QT]);
        Dfeat = reshape(f.', 1, []);
        row = row + 1;
        labeled_dataset(row,:) = [Dfeat, x];
    end

    % store convergence column for this sample
    convergence_data(:, col+1) = current_results;

    if mod(col,5) == 0   %Print progress every 5 samples using modulo
        fprintf('Progress: sample %d / %d  (Current Load_Variation = %.4f)\n', col, num_samples, Load_Variation);
    end
end

% Trim to the number of rows actually collected
labeled_dataset = labeled_dataset(1:row,:);

Column_Labels = "";
for j = otl
    Column_Labels = strcat(Column_Labels,sprintf('PF Line %d, ',j));
    Column_Labels = strcat(Column_Labels,sprintf('QF Line %d, ',j));
    Column_Labels = strcat(Column_Labels,sprintf('PT Line %d, ',j));
    Column_Labels = strcat(Column_Labels,sprintf('QT Line %d, ',j));
end
Column_Labels = strcat(Column_Labels,sprintf('Label'));

convFile = sprintf('Convergence_Data_%s_%s_%s.csv', system, label, sol_type);
dataFile = sprintf('Branchdata_%s_%s_%s.csv',      system, label, sol_type);

% Convergence Data CSV
writematrix(convergence_data, convFile);

% Branch Data CSV
fid = fopen(dataFile, 'w');
if fid == -1, error('Cannot open file %s', dataFile); end
fprintf(fid, '%s', Column_Labels);
fclose(fid);
writematrix(labeled_dataset, dataFile, 'WriteMode', 'append');


%% Save RNG Seed and Other User Paramters
duration = toc;
fprintf('Total Time: %.2f seconds\n', duration);

rng_seeds = table(system,string(label), seed, samples, rand_var, string(sol_type), duration, ...
    'VariableNames', {'case','data_label','rng_seed','samples_per_scenario','percent_load_var(%)','DC or AC','Total Time'});

if exist('RNG_Seeds.csv','file')
    writetable(rng_seeds, 'RNG_Seeds.csv', 'WriteMode', 'append');
else
    writetable(rng_seeds, 'RNG_Seeds.csv');
end
end