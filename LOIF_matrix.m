%Run LOIF_matrix(case file name) --> Ex. LOIF_matrix('case118')

function LOIF_matrix(system)
mpc = loadcase(system); %
mpc = ext2int(mpc);
define_constants
% slack = 7049;
slack = mpc.bus(mpc.bus(:,2) == 3, 1);    % returns bus number(s)
display(slack)
PTDF = makePTDF(mpc,slack);

s_base = mpc.baseMVA; %Base MVA for entire system

LODF = makeLODF(mpc.branch,PTDF);

results = rundcopf(mpc);
Pre_branchdata = results.branch(:,PF);
LODF_MW = LODF; %LODF in terms of MW

for l= 1:size(mpc.branch,1)
    for k=1:size(mpc.branch,1)
    LODF_MW(l,k) = Pre_branchdata(k) * LODF_MW(l,k);
    end
end

LOIF = LODF_MW;
LOIF_pu = LODF_MW;
LOIF_change = LODF_MW;
for i=1:size(mpc.branch,1) %for every row
    for j=1:size(mpc.branch,1) %go through each column
               LOIF(i,j) = LODF_MW(i,j) + Pre_branchdata(i);
               LOIF_pu(i,j) = LODF_MW(i,j) + Pre_branchdata(i);
               LOIF_pu(i,j) = LOIF_pu(i,j) / s_base;
               LOIF_change(i,j) = LODF_MW(i,j)/Pre_branchdata(i);
          
    end
end
row_labels = "";
column_labels = "";
for j=1:size(mpc.branch,1)
    row_labels = strcat(row_labels,sprintf('line%d',j));
    column_labels = strcat(column_labels,sprintf('outage%d,',j));
end

lodf_file = sprintf('LODFmatrix_%s.csv',system);
csvwrite(lodf_file, LODF);
% Read the contents of the specified file
S = fileread(lodf_file);
% Open the file for writing, and check for successful opening
FID = fopen(lodf_file, 'w');
if FID == -1, error('Cannot open file %s', lodf_file); end
% Write column labels to the file
fprintf(FID, "%s\n", column_labels);
% Write the contents read from the file back to it
fprintf(FID, "%s", S);
% Close the file
fclose(FID);


loif_file = sprintf('LOIFmatrix_%s.csv',system);
csvwrite(loif_file, LOIF_change);
% Read the contents of the specified file
S = fileread(loif_file);
% Open the file for writing, and check for successful opening
FID = fopen(loif_file, 'w');
if FID == -1, error('Cannot open file %s', loif_file); end
% Write column labels to the file
fprintf(FID, "%s\n", column_labels);
% Write the contents read from the file back to it
fprintf(FID, "%s", S);
% Close the file
fclose(FID);
end