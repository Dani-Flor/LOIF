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
    %% Setup argument default values where necessary
    % (A) 8 Parameters are given except: otl,
    %      default: All lines
    %
    % (B) 7 Parameters are given except: sol_type & otl
    %      default: AC, All lines
    %
    % (C) 6 Parameters are given except: rng_seed,sol_type, & otl
    %      defult:POSIX Time, AC, All lines
    %
    % (D) 5 Parameters are given execpt: rand_var,rng_seed,sol_type,otl
    %      default: 5%,POSIX Time, AC, All lines
    %
    %  (E) 4 Parameters are given except: hourly_var, rand_var, rng_seed, sol_type, otl
    %      default: 1, 5%, POSIX Time, AC, All lines
    %
    % (F) 3 Paramters are given except: output_dir, hourly_var, rand_var, rng_seed, sol_type,otl
    %      default: Current Directory, 1, 5, POSIX Time, AC, All lines
    %
    % (G) ERROR if User does  not provide main parameters: system, label, samples
    if nargin == 8                          %(A)
        otl = [];
    elseif nargin == 7                      %(B)
        otl = [];
        sol_type = "AC";
    elseif nargin == 6                      %(C)
        otl = [];
        sol_type = "AC";
        rng_seed = 0;
    elseif nargin == 5                      %(D)
        otl = [];
        sol_type = "AC";
        rng_seed = 0;
        rand_var = 5;
    elseif nargin == 4                      %(E)
        otl = [];
        sol_type = "AC";
        rng_seed = 0;
        rand_var = 5;
        hourly_var = 1.0;
    elseif nargin == 3                      %(F)
        otl = [];
        sol_type = "AC";
        rng_seed = 0;
        rand_var = 5;
        hourly_var = 1.0;
        output_dir = "./";
    elseif nargin < 3                       %(G)
        error('User Must Specify: (1) Case System: Ex. "case118" (2) Label: Ex. "demo" (3) Number of Samples: Ex. 50')
    end

    %
    % Setup output directory
    %
    %% Make sure output directory string ends with a slash
    if output_dir(end) ~= '/'
      output_dir = sprintf('%s/', output_dir);
    end
    %% If directory does not exist, create it
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end


    %% Calculate LOIF and LODF Matrices
    LOIF_matrix(system,output_dir)

    %% Initialization
    % (A) Start Clock
    % (B) Load MATPOWER indices (PD, QD, BR_STATUS, PF, QF, PT, QT)
    % (C) Hide Warning Messages
    % (D) Disable MATPOWER output messages
    % (E) Create Copy of System
    % (F) Calculate the number of lines
    % (G) Check if Sol. Type is AC or DC, ERROR if not
    % (H) Check if user wants all lines or if otls provided by user are valid
    % (I) Set RNG Seed Provided By User or set RNG seed with POSIX Time

    tic;                                                               %(A)
    define_constants;                                                  %(B)
    warning('off','all');                                              %(C)
    mpopt = mpoption('verbose',0,'out.all',0);                         %(D)
    mpc_copy = loadcase(system);                                       %(E)
    num_lines  = size(mpc_copy.branch,1);                              %(F)

    if (strcmp(sol_type,'AC')) | (strcmp(sol_type,'DC'))               %(G)
    else
        error('User Must Type "DC" or "AC"')
    end

    if isempty(otl)                                                    %(H)
        otl = 1:num_lines;
    elseif ~all(ismember(otl,1:num_lines))
        error('System only has %d lines. ML contains out-of-range entries.', num_lines);
    end

    if rng_seed == 0                                                   %(I)
        rng_seed = (now - 719529) * 86400;
    end
    rng(rng_seed);

    %% Initialize Variables and Datasets
    % (A) Calculate # of rows needed: one for each line + row for
    % (B) Calculate # of columns needed: 4 for each OTL [PF,QF,PT,QF] + Label Column
    % (C) Create Empty Dataset for Labeled Data/Training Data
    % (D) Initialize Row Index (starting with 0)
    % (E) Create Empty Dataset for Convergence Data
    % (F) Enter Index Column to Convergence Dataset (All Scenarios: Normal Conditions + All Outages Scenarios
    % (G) Create Array To Hold Convergence Results for all scenarios from previous sample/iteration
    % (H) Create a copy of Active Loads in system
    % (I) Create a copy of Reactive Loads in system
    num_rows  = num_lines + 1;                                         %(A)
    num_col  = numel(otl)*4 + 1;                                       %(B)

    labeled_dataset = zeros(samples*num_rows, num_col);                %(C)
    row = 0;                                                           %(D)

    % convergence data
    convergence_data = zeros(num_rows, samples+1);                     %(E)
    convergence_data(:,1) = 0:num_lines;                               %(F)

    previous_results = ones(num_rows,1);                               %(G)

    PD_copy = mpc_copy.bus(:, PD);                                     %(H)
    QD_copy = mpc_copy.bus(:, QD);                                     %(I)

    %
    % Collect each sample (i) for each line outage condition (x)
    %
    for i = 0:(samples-1)

        col = i + 1;    %update Column Index (used for convergence dataset)

        %% Calculate Load Variation (Hourly + Random)
        % (A) IF on first sample, no random load variation
        % (B) ELSE Calculate Random Variation From -rand_var% to +rand_var%
        % (C) Find Total Load Variation: Hourly Variation + Random Variation
        % (D) Add Variation to Active Loads
        % (E) Add Variation to Reactive Loads
        % (F) Use copy of System (No Load Variation Yet)
        % (G) Update Active Loads (with variation)
        % (H) Update Reactive Loads (with variation)
        % (I)  Initialize an array for current sample's convergence results
        if i == 0                                                      %(A)
            Load_Variation = 0;
        else                                                           %(B)
            percent_var = rand_var * 0.01;
            a = -percent_var;
            b = percent_var;
            Load_Variation = (b-a) * rand() + a;
        end

        v = hourly_var * (1 + Load_Variation);                         %(C)
        PDs   = PD_copy * v;                                           %(D)
        QDs   = QD_copy * v;                                           %(E)
        mpc_updated = mpc_copy;                                        %(F)
        mpc_updated.bus(:, PD) = PDs;                                  %(G)
        mpc_updated.bus(:, QD) = QDs;                                  %(H)

        current_results = zeros(num_rows,1);                           %(I)
    %% For Normal Conditions Run OPF solution and collect Active/Reactive Power at all lines and collect generator data
    % (A) Check If Previous Sample Converged For Normal Conditions
    % (B) Update System With Load Variation
    % (C) Check Solution Type
    % (D) IF AC run AC OPF Solution
    % (E) ELSE run DC OPF Solution
    % (F) Check if Current Solution Converged
    % (G) If Solution Converged Collect Generator Data (Used During Outage Scenarios)
    % (H) IF Solution Converged Mark Current Result With 1
    % (I) IF Solution Converged Collect Power Flow Measurements [PF,QF,PT,QT] for OTLs
    % (J) Change Vector To Array
    % (K) Update Row Index
    % (L) Add Power Flow Measurements (Features) + Label to Labeled Dataset
    % (M) IF Current Solution Failed Mark Previous Results with 0 (Used For Next Sample)
    % (N) IF Current Solution Failed Mark Current Results with 0
    % (O) Store Current Convergence Results to Convergence Dataset
    % (P) Skip Outage Scenarios if Normal Conditions Failed
    % (Q) ELSE: Normal Conditions Previously Failed
    % (R) Mark Previous Result As Failed with 0
    % (S) Mark Current Result As Failed  with 0
    % (T) Store Current Results into Convergennce Dataset
    % (U) Skip Outage Scenarios if Normal Condition Failed Previously

        if previous_results(1) == 1                                    %(A)
            mpc_norm = mpc_updated;                                    %(B)

            if strcmp(sol_type,'AC')                                   %(C)
                results0 = runopf(mpc_norm, mpopt);                    %(D)
            else
                results0 = rundcopf(mpc_norm, mpopt);                  %(E)
            end

            if results0.success == 1                                   %(F)
                gen_dat = results0.gen;                                %(G)
                current_results(1) = 1;                                %(H)
                f = results0.branch(otl, [PF QF PT QT]);               %(I)
                Dfeat = reshape(f.', 1, []);                           %(J)
                row = row + 1;                                         %(K)
                labeled_dataset(row,:) = [Dfeat, 0];                   %(L)
            else
                previous_results(1) = 0;                               %(M)
                current_results(1) = 0;                                %(N)
                convergence_data(:, col+1) = current_results;          %(O)
                continue;                                              %(P)
            end
        else                                                           %(Q)
            previous_results(1) = 0;                                   %(R)
            current_results(1) = 0;                                    %(S)
            convergence_data(:, col+1) = current_results;              %(T)
            continue;                                                  %(U)
        end

    %% For Each Outage Scenario run PF solution with Fixed Generation (From OPF solution), collect power flows for each otl
    % (A) Before Anything Check If Current Outage Scenario Previously Converged
    % (B) IF Previously Failed Mark Current Result with 0 (Skip Forever)
    % (C) IF Previously Failed Skip PF Run and Feature Collection
    % (D) Update System With Load Variation
    % (E) Fix Generation of System With OPF Results For Normal Conditions
    % (F) Change Branch Status of line to 0 (Simulate Outage)
    % (G)  Check Solution Type
    % (H)  IF AC run AC PF Solution
    % (I)  ELSE run DC PF Solution
    % (J) Check If Solution Failed To Converge
    % (K) IF Failed Mark Previous Result with 0 (Used For Next Iteration)
    % (L) IF Failed Mark Current Result with 0
    % (M) IF Failed Skip Feature Collection
    % (N) ELSE Mark Current Results as Success with 1
    % (O) Collect Active/Reactive Power for OTLs
    % (P) Change Vector to Array
    % (Q) Update Row Index
    % (R) Add Power Flow Measurements (Features) + Label to Labeled Dataset
    % (S) Store Current Sample's Convergence Results to Convergence Dataset
    % (T) Print Progress Every 5 Samples
        for x = 1:num_lines    %For Each Transmission Line
            if previous_results(x+1) == 0                              %(A)
                current_results(x+1) = 0;                              %(B)
                continue;                                              %(C)
            end

            mpc_out = mpc_updated;                                     %(D)
            mpc_out.gen = gen_dat;                                     %(E)
            mpc_out.branch(x, BR_STATUS) = 0;                          %(F)

            if strcmp(sol_type,'AC')                                   %(G)
                results = runpf(mpc_out, mpopt);                       %(H)
            else
                results = rundcpf(mpc_out, mpopt);                     %(I)
            end

            if results.success ~= 1                                    %(J)
                previous_results(x+1) = 0;                             %(K)
                current_results(x+1) = 0;                              %(L)
                continue;                                              %(M)
            else
                current_results(x+1) = 1;                              %(N)
            end

            f = results.branch(otl, [PF QF PT QT]);                    %(O)
            Dfeat = reshape(f.', 1, []);                               %(P)
            row = row + 1;                                             %(Q)
            labeled_dataset(row,:) = [Dfeat, x];                       %(R)
        end
        convergence_data(:, col+1) = current_results;                  %(S)
        if mod(col,5) == 0                                             %(T)
            fprintf(['Progress: sample %d / %d  ' ...
                '(Current Load_Variation = %.4f)\n'], ...
                col, samples, Load_Variation);
        end
    end

    %
    % Sample collection completed, print out total compuation time
    %
    duration = toc;  %Stop Clock
    fprintf('Total Time: %.2f seconds\n', duration);


    %% Generate output files: convergence, training data, and user parameters
    % (A) Let user know that CSV files are being created
    % (B) Create File Name For Convergence Data CSV
    % (C) Create File Name For Training Data CSV
    % (D) Create File Name For User Parameters CSV
    % (E) Create Column Labels: Power Flow [PF,QF,PT,QT] per OTL + Label
    % (F) Write Covergence Results To CSV File
    % (G) Trim Unsused Rows From Training Data
    % (H) Write Training Data To CSV File
    fprintf('Writing output files: ');                                 %(A)
    convFile = sprintf('%s%s_convergence_%s_%s.csv',  output_dir, ...  %(B)
        label, system, sol_type);
    dataFile = sprintf('%s%s_trainingdata_%s_%s.csv', output_dir, ...  %(C)
        label, system, sol_type);
    paramFile = sprintf('%suser_parameters.csv', output_dir);          %(D)
    column_labels = "";                                                %(E)
    for j = otl
        column_labels = strcat(column_labels,sprintf('PF Line %d, ',j));
        column_labels = strcat(column_labels,sprintf('QF Line %d, ',j));
        column_labels = strcat(column_labels,sprintf('PT Line %d, ',j));
        column_labels = strcat(column_labels,sprintf('QT Line %d, ',j));
    end
    column_labels = strcat(column_labels,sprintf('Label'));
    csvwrite(convFile, convergence_data); % csvwrite()                 %(F)
    labeled_dataset = labeled_dataset(1:row,:);                        %(G)
    csvwrite(dataFile,labeled_dataset);                                %(H)

    %% ADD COLUMN LABELS TO TRAINING DATA CSV
    % (A)  Store a copy of the data from csv file
    % (B)  Open csv (overwrite), Send ERROR if not able to
    % (C)  Insert column labels at the beginning of the CSV file
    % (D)  Store copy back into CSV
    % (E)  Close CSV File

    S = fileread(dataFile);                                            %(A)

    FID = fopen(dataFile, 'w');                                        %(B)

    if FID == -1, error('Cannot open file %s', dataFile); end          %(B)

    fprintf(FID, "%s\n", column_labels);                               %(C)

    fprintf(FID, "%s", S);                                             %(D)

    fclose(FID);                                                       %(E)

    %% Save RNG seed, other user parameters and data to CSV
    %  (A)  IF CSV user parameter file already exists, append to it,
    %  (B)  Open CSV (append mode), Send ERROR if not able to
    %  (C)  ELSE create CSV(overwrite mode), Send ERROR if not able to
    %  (D)  Create column labels
    %  (E)  Add column labels to CSV
    %  (F)  Write parameter values to CSV file
    %  (G)  Print User Parameters to CSV
    %  (H)  Close CSV

    if exist(paramFile,'file')                                         %(A)
        FID = fopen(paramFile,'a');                                    %(B)
        if FID==-1, error('Cannot open file %s', paramFile); end
    else                                                               %(C)
        FID = fopen(paramFile,'w');
        if FID==-1, error('Cannot open file %s', paramFile); end
        column_labels = "label,system,samples,hourly_var,rand_var,rng_seed,sol_type,run_time";     %(D)
        fprintf(FID, '%s\n', column_labels);                           %(E)
    end

    fmt = '%s,%s,%d,%.6f,%.6f,%.6f,%s,%d\n';                           %(F)

    fprintf(FID, fmt, char(label), char(system), samples, ...          %(G)
        hourly_var, rand_var, rng_seed, char(sol_type), duration);
    fclose(FID);                                                       %(H)

    %
    % Let User Know That CSV Files are Saved
    %
    fprintf('done.\n');

end % end of lod_labeled_datagen()
