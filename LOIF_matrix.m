%Run LOIF_matrix(case file name) --> Ex. LOIF_matrix('case118')

function LOIF_matrix(mpc_string)
mpc = loadcase(mpc_string); %
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

all_rows = [];
all_columns = [];
for j=1:size(mpc.branch,1)
    all_columns = cat(2,all_columns,string(sprintf('outage%d',j)));
    all_rows = cat(2,all_rows,string(sprintf('line%d',j)));
end

table = array2table(LOIF_change);
table.Properties.RowNames = all_rows;
table.Properties.VariableNames = all_columns;

writetable(table,sprintf('LOIFmatrix_%s.csv',mpc_string))
end