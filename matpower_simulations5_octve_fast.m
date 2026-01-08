function matpower_simulations5_octve_fast(mpc_string,ML,hourly_load_variation,percent_load_var,samples_per_scenario,dataset_label,rng_seed,DC_or_AC)
% Faster version of matpower_simulations5_octve()

LOIF_matrix(mpc_string)
tic;

define_constants;                 % MATPOWER indices (PD,QD,BR_STATUS,PF,...)
warning('off','all');             
mpopt = mpoption('verbose',0,'out.all',0);   

if (strcmp(DC_or_AC,'AC')) | (strcmp(DC_or_AC,'DC'))
else
    error('User Must Type "DC" or "AC"')
end

% Load base case only once
mpc_copy = loadcase(mpc_string); %Copy of the System before any changes are made
num_lines  = size(mpc_copy.branch,1);
num_buses  = size(mpc_copy.bus,1);

% Monitor line list
if isempty(ML)   % sentinel => all lines
    ML = 1:num_lines;
elseif ~all(ismember(ML,1:num_lines))
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

%% --- Preallocate outputs ---
num_samples     = samples_per_scenario + 1;     % i = 0..samples_per_scenario
num_scenarios  = num_lines + 1;                       % x = 0..nL
num_features  = numel(ML)*4 + 1;              % 4 per monitor line + label

% D: we preallocate max rows; then trim to actual rows collected
D = zeros(num_samples*num_scenarios, num_features);
% display(D)
row = 0;  %used as counter

% R -> convergence matrix
R = zeros(num_scenarios, num_samples+1);
R(:,1) = 0:num_lines;
% display(R)
Result_Holder = ones(num_scenarios,1);         %Results of All outages and normal conditions
display(Result_Holder)

%Orignal PD/QD
PD_copy = mpc_copy.bus(:, PD);
QD_copy = mpc_copy.bus(:, QD);
% 
% %% --- Main loops ---

for i = 0:samples_per_scenario
    it = i + 1;
    % Random load variation for this sample
    if i == 0
        Load_Variation = 0;
    else
        percent_var = percent_load_var * 0.01;
        a = -percent_var;
        b = percent_var;
        Load_Variation = (b-a) * rand() + a;  %This gives a random load variation between -percent: +percent
    end

    %     % Change the Real and Reactive Demand at each bus
    % for j=1:size(mpc.bus,1) %118 buses
    %     mpc.bus(j, PD) = (mpc.bus(j,PD)*hourly_load_variation) + (mpc.bus(j,PD)*hourly_load_variation*Load_Variation);
    %     mpc.bus(j, QD) = (mpc.bus(j,QD)*hourly_load_variation) + (mpc.bus(j,QD)*hourly_load_variation*Load_Variation);
    % end

    % Vector Format Instead of looping through every bus
    scale = hourly_load_variation * (1 + Load_Variation);
    PDs   = PD_copy * scale;
    QDs   = QD_copy * scale;

    mpc_base = mpc_copy;  %Copy original system (No changes)
    mpc_base.bus(:, PD) = PDs;    %Update Active Demand with Random Load Variation
    mpc_base.bus(:, QD) = QDs;    %Update Reactive Demand with Random Load Variation

    % Convergence results for current sample
    R1 = zeros(num_scenarios,1);
%% Collect Data For Base Load Conditions
    if Result_Holder(1) == 1
        mpc = mpc_base;

        if strcmp(DC_or_AC,'AC')
            results0 = runopf(mpc, mpopt);  %AC OPF
        else
            results0 = rundcopf(mpc, mpopt); %DC OPF
        end

        if results0.success == 1
            gen_dat = results0.gen;  %Collect Gen Data(Normal conditions) to fix generation for outage conditions
            R1(1) = 1;     %Previous Result for Normal Conditions

            % Collect features
            B = results0.branch(ML, [PF QF PT QT]);    % (numel(ML) x 4)
            Dfeat = reshape(B.', 1, []);
            label = 0;

            row = row + 1;
            D(row,:) = [Dfeat, label];
        else
            % If x=0 fails, mark and skip all outages for this sample
            Result_Holder(1) = 0;
            R1(1) = 0;
            % no gen_dat available -> skip remaining x
            R(:, it+1) = R1;
            continue;
        end
    else
        % previously failed, always skip
        R1(1) = 0;
        R(:, it+1) = R1; %save results for in convergence matrix
        continue;
    end

    %% For All Possible Outages
    for x = 1:num_lines
        if Result_Holder(x+1) == 0
            % previously failed => skip forever
            R1(x+1) = 0;
            continue;
        end

        mpc = mpc_base;
        mpc.branch(x, BR_STATUS) = 0;     % disconnect line x
        mpc.gen = gen_dat;               % fixed generation from normal OPF

        if strcmp(DC_or_AC,'AC')
            results = runpf(mpc, mpopt);  %AC power flow
        else
            results = rundcpf(mpc, mpopt);  %DC power flow
        end

        if results.success ~= 1
            % mark as failed if already failed once
            Result_Holder(x+1) = 0;
            R1(x+1) = 0;
            continue;  %skip feature collection
        end

        R1(x+1) = 1;

        % Collect features
        B = results.branch(ML, [PF QF PT QT]);
        Dfeat = reshape(B.', 1, []);
        label = x;

        row = row + 1;
        D(row,:) = [Dfeat, label];
    end

    % store convergence column for this sample
    R(:, it+1) = R1;

    %
    if mod(it,5) == 0   %Print progress every 5 samples using modulo
        fprintf('Progress: sample %d / %d  (Load_Variation = %.4f)\n', it, num_samples, Load_Variation);
    end
end

% Trim D to the number of rows actually collected
D = D(1:row,:);

%% --- Column labels (build once) ---
Column_Labels = "";
for j = ML
    Column_Labels = strcat(Column_Labels,sprintf('PF Line %d, ',j));
    Column_Labels = strcat(Column_Labels,sprintf('QF Line %d, ',j));
    Column_Labels = strcat(Column_Labels,sprintf('PT Line %d, ',j));
    Column_Labels = strcat(Column_Labels,sprintf('QT Line %d, ',j));
end
Column_Labels = strcat(Column_Labels,sprintf('Label'));

convFile = sprintf('Convergence_Data_%s_%s_%s.csv', mpc_string, dataset_label, DC_or_AC);
dataFile = sprintf('Branchdata_%s_%s_%s.csv',      mpc_string, dataset_label, DC_or_AC);

% Convergence Data CSV
writematrix(R, convFile);

% Branch Data CSV
fid = fopen(dataFile, 'w');
if fid == -1, error('Cannot open file %s', dataFile); end
fprintf(fid, '%s', Column_Labels);
fclose(fid);
writematrix(D, dataFile, 'WriteMode', 'append');


%% Save RNG Seed and Other User Paramters
duration = toc;
fprintf('Total Time: %.2f seconds\n', duration);

rng_seeds = table(mpc_string,string(dataset_label), seed, samples_per_scenario, percent_load_var, string(DC_or_AC), duration, ...
    'VariableNames', {'case','data_label','rng_seed','samples_per_scenario','percent_load_var(%)','DC or AC','Total Time'});

if exist('RNG_Seeds.csv','file')
    writetable(rng_seeds, 'RNG_Seeds.csv', 'WriteMode', 'append');
else
    writetable(rng_seeds, 'RNG_Seeds.csv');
end

end
