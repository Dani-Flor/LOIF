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

column_labels = "";
for j=1:size(mpc.branch,1)
column_labels = strcat(column_labels,sprintf('outage%d, ',j));
end

lodf_file = sprintf('LODFmatrix_%s.csv',system);
loif_file = sprintf('LOIFmatrix_%s.csv',system);

fid = fopen(lodf_file, 'w'); %open labeled dataset CSV file
if fid == -1, error('Cannot open file %s', dataFile); end  %ERROR, could not open
fprintf(fid, '%s', column_labels);  %Save Column Labels first
fclose(fid);
writematrix(LODF, lodf_file, 'WriteMode', 'append');

fid = fopen(loif_file, 'w'); %open labeled dataset CSV file
if fid == -1, error('Cannot open file %s', dataFile); end  %ERROR, could not open
fprintf(fid, '%s', column_labels);  %Save Column Labels first
fclose(fid);
writematrix(LOIF_change, loif_file, 'WriteMode', 'append');
% table = array2table(LOIF_change);
% table.Properties.RowNames = all_rows;
% table.Properties.VariableNames = column_labels;
% 
% writetable(table,sprintf('LOIFmatrix_%s.csv',system))
% 
% table = array2table(LODF);
% table.Properties.RowNames = all_rows;
% table.Properties.VariableNames = column_labels;
% writetable(table, sprintf('LODFmatrix_%s.csv', system));
end