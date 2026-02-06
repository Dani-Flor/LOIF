% function lod_labeled_datagen(label,system,samples,output_dir,hourly_var,rand_var,rng_seed,sol_type,otl)
% label: Used to identify file name of Training Data
% system: Name of the case file in matpower that we are interested in (Ex. 'case118' -> 118 bus system)
% samples: Number of Samples For Each Scenario (Outages and Normal Conditions)
% output_dir: Directory to write output files
% hourly_var: Hourly Load Variation will be used later when we try to implement a daily load profile.
% rand_var: Percentage that represents load variation (random number from -% to +%).
% rng_seed: The seed of Random Number Generator (used for random load variation).
%     If we set rng_seed to 0, seed is Generated using POSIX time (number of seconds since 1970)
%     If we want to replicate exact load variation, we enter the RNG seed of previous training data.
% sol_type: This is to determine if the user wants to collect results by running AC or DC solutions.
% otl: This is the monitored line(s) which can be a single integer, an array of integers, or the sentinel value '[]' (all lines).
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

function lod_labeled_datagen(label,system,samples,output_dir,hourly_var,rand_var,rng_seed,sol_type,otl)
    %
    % Setup argument default values where necessary
    %
    if nargin == 8
        otl = [];
    elseif nargin == 7
        otl = [];
        sol_type = "AC";
    elseif nargin == 6
        otl = [];
        sol_type = "AC";
        rng_seed = 0;
    elseif nargin == 5
        otl = [];
        sol_type = "AC";
        rng_seed = 0;
        rand_var = 5;
    elseif nargin == 4
        otl = [];
        sol_type = "AC";
        rng_seed = 0;
        rand_var = 5;
        hourly_var = 1.0;
    elseif nargin == 3
        otl = [];
        sol_type = "AC";
        rng_seed = 0;
        rand_var = 5;
        hourly_var = 1.0;
        output_dir = "./"
    end

    %
    % Setup output directory
    %
    %% Make sure output directory string ends with a slash
    if output_dir(end) != '/'
      output_dir = sprintf('%s/', output_dir);
    end
    %% If directory does not exist, create it
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end


    %% Calculate LOIF and LODF Matrices
    LOIF_matrix(system,output_dir)

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
        %calculate POSIX Time manually
        rng_seed = (now - 719529) * 86400;  %now function is compatible with both
        %MATLAB and Octave: (serial number of current date - the serial number
        %                                    Jan 1,1970) * number of seconds in a day
    end
    rng(rng_seed);

    %% Initialize Variables
    num_samples = samples;
    num_rows  = num_lines + 1;
    num_col  = numel(otl)*4 + 1;              % 4 features per line([PF,QF,PT,QF]) + label

    labeled_dataset = zeros(num_samples*num_rows, num_col);
    row = 0;  %used as counter

    % convergence data
    convergence_data = zeros(num_rows, num_samples+1);
    convergence_data(:,1) = 0:num_lines;

    previous_results = ones(num_rows,1);         %Convergence Results of All outages and normal condition of prev. sample

    %Orignal PD/QD
    PD_copy = mpc_copy.bus(:, PD); %Copy of Active Loads
    QD_copy = mpc_copy.bus(:, QD); %Copy of Reactive Loads

    %
    % Collect each sample (i) for each line outage condition (x)
    %
    for i = 0:(samples-1)

        col = i + 1;

        % Random load variation for current sample
        if i == 0   %If i = 0, collect data with base load (no variation)
            Load_Variation = 0;
        else
            percent_var = rand_var * 0.01;
            a = -percent_var;
            b = percent_var;
            Load_Variation = (b-a) * rand() + a;  %This gives a random load variation between -percent: +percent
        end

        v = hourly_var * (1 + Load_Variation); %Find total variation
        PDs   = PD_copy * v; %update Active load with variation
        QDs   = QD_copy * v; %update Reactive load with variation
        mpc_updated = mpc_copy;  %Copy original system (No changes)
        mpc_updated.bus(:, PD) = PDs;    %Update Active Demand with Load Variation
        mpc_updated.bus(:, QD) = QDs;    %Update Reactive Demand with Load Variation

        % Convergence results for current sample
        current_results = zeros(num_rows,1);
        %% For Normal Conditions Run OPF solution and collect Active/Reactive Power at all lines and collect generator data
        if previous_results(1) == 1  %If previous sample converged for normal conditions
            mpc_norm = mpc_updated;  %Load system with load variation

            if strcmp(sol_type,'AC')
                results0 = runopf(mpc_norm, mpopt);   %AC OPF
            else
                results0 = rundcopf(mpc_norm, mpopt); %DC OPF
            end

            if results0.success == 1
                gen_dat = results0.gen;  %Collect Gen Data(Normal conditions) to fix generation for outage conditions
                current_results(1) = 1;  %Save Convergence Result

                f = results0.branch(otl, [PF QF PT QT]); %Collect Active/Reactive Power for OTLs
                Dfeat = reshape(f.', 1, []); %vector to array

                row = row + 1;
                labeled_dataset(row,:) = [Dfeat, 0]; % Save Features and Label to labeled dataset
            else
                %If normal condition failed previously
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

        %% For Each Outage Scenario run PF solution with Fixed Generation (From OPF solution), collect power flows for each otl
        for x = 1:num_lines

            %Check if previous sample converged or not.
            if previous_results(x+1) == 0
                % previously failed => skip forever
                current_results(x+1) = 0;
                continue;
            end

            %% Load case, fix generation then disconnect line. After that run PF solution
            mpc_out = mpc_updated;               % load system with load variation
            mpc_out.gen = gen_dat;               % fixed generation from normal OPF
            mpc_out.branch(x, BR_STATUS) = 0;    % disconnect line x

            %Check if solution type is AC or DC. Then run powerflow (PF) solution
            if strcmp(sol_type,'AC')
                results = runpf(mpc_out, mpopt);  %AC power flow solution
            else
                results = rundcpf(mpc_out, mpopt);  %DC power flow solution
            end

            %% Check convergence for current outage, if failed skip power flow collection.
            % If succeeded, collect powerflows at OTLs and create Label for
            % outage

            %Check if outage converged/not converge for current sample
            if results.success ~= 1 %check if failed
                previous_results(x+1) = 0; %update convergence results
                current_results(x+1) = 0; %update convergence results
                continue;  %skip feature collection
            end

            current_results(x+1) = 1; %update if solution converged


            f = results.branch(otl, [PF QF PT QT]); %Collect Active/Reactive Power for OTLs
            Dfeat = reshape(f.', 1, []); %vector to array

            row = row + 1;
            labeled_dataset(row,:) = [Dfeat, x]; % Save Features and Label to labeled dataset
        end

        % store current sample's convergence results in convergence dataset.
        convergence_data(:, col+1) = current_results;

        if mod(col,5) == 0   %Print progress every 5 samples using modulo
            fprintf('Progress: sample %d / %d  (Current Load_Variation = %.4f)\n', col, num_samples, Load_Variation);
        end
    end

    %
    % Sample collection completed, print out total compuation time
    %
    duration = toc;  %Collect Computation Time
    fprintf('Total Time: %.2f seconds\n', duration);


    %
    % Generate output files: convergence, training data, and user parameters
    %
    fprintf('Writing output files: ');
    %% Create CSV file names for Convergence Results, Labeled Dataset, and User Parameters
    convFile = sprintf('%s%s_convergence_%s_%s.csv',  output_dir, label, system, sol_type); % Name of file for convergence results
    dataFile = sprintf('%s%s_trainingdata_%s_%s.csv', output_dir, label, system, sol_type); % Name of file for labeled dataset
    paramFile = sprintf('%suser_parameters.csv', output_dir);

    %% Create Column Labels for Labeled Dataset (4 features per OTL and 1 Label)
    column_labels = "";
    for j = otl
        column_labels = strcat(column_labels,sprintf('PF Line %d, ',j));
        column_labels = strcat(column_labels,sprintf('QF Line %d, ',j));
        column_labels = strcat(column_labels,sprintf('PT Line %d, ',j));
        column_labels = strcat(column_labels,sprintf('QT Line %d, ',j));
    end
    column_labels = strcat(column_labels,sprintf('Label'));

    %
    % Write convergence results to CSV
    %
    csvwrite(convFile, convergence_data); % csvwrite() does not include column labels

    %
    % Write power measurement features and line outage condition labels to CSV
    %
    labeled_dataset = labeled_dataset(1:row,:); % Trim to the number of rows actually collected
    csvwrite(dataFile,labeled_dataset);         % csvwrite() does not include column labels
    %
    % Insert column labels at the beginning of the CSV file
    %
    S = fileread(dataFile);
    FID = fopen(dataFile, 'w');
    if FID == -1, error('Cannot open file %s', dataFile); end
    % Write column labels to the file
    fprintf(FID, "%s\n", column_labels);
    % Write the contents read from the file back to it
    fprintf(FID, "%s", S);
    % Close the file
    fclose(FID);

    %
    % Save RNG seed, other user parameters and data to CSV
    %
    % If CSV user parameter file already exists, append to it, otherwise create new CSV file
    if exist(paramFile,'file')
        FID = fopen(paramFile,'a');
        if FID==-1, error('Cannot open file %s', paramFile); end
    else
        FID = fopen(paramFile,'w');
        if FID==-1, error('Cannot open file %s', paramFile); end
        column_labels = "label,system,samples,hourly_var,rand_var,rng_seed,sol_type,run_time";
        fprintf(FID, '%s\n', column_labels);
    end
    % Write parameter values to CSV file
    fmt = '%s,%s,%d,%.6f,%.6f,%.6f,%s,%d\n';
    % ensure label and system are char (use char(...) or string(...))
    fprintf(FID, fmt, char(label), char(system), samples, hourly_var, rand_var, rng_seed, char(sol_type), duration);
    fclose(FID);
    fprintf('done.\n');

end % end of lod_labeled_datagen()
