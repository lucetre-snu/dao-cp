
clc;clear;close all;
addpath(genpath('../'));

%% generate data
dims = [20, 20, 20, 200];
N = length(dims);
tao = round(0.2*dims(end));
TT = dims(end)-tao;
R = 5;
% X = generateData(dims, R, 20);
% save X.mat X

load X.mat;
whos X;

Uest = cpd(X, R, 'Algorithm', @cpd_als);
Uest_N = Uest{end};
T = cpdgen(Uest);
norm(tensor(X)-tensor(T))

% for i = tao-5:tao
%     fprintf('>> Uest_N(%d) ', i); 
%     disp(Uest_N(i,:));
% end
% for i = 1:20
%     fprintf('>> Uest_1(%d) ', i); 
%     disp(Uest{1}(i,:));
% end

% whos T;
% for i = tao-5:tao
%     fprintf('>> X(%d) ', i); 
%     disp(X(20,20,20,i));
% end

%% initialization
% get initX
idx = repmat({':'}, 1, length(dims));
idx(end) = {1:tao};
initX = X(idx{:});

initAs = cpd(initX, R, 'Algorithm', @cpd_als);

% T1 = full(ktensor(initAs));
% whos T1;
% norm(tensor(initX)-T1)

% initialize onlineCP method
[onlinePs, onlineQs] = onlineCP_initial_tenlab(initX, initAs, R);
onlineAs = initAs(1:end-1);
onlineAs_N = initAs{end};
for i = tao-5:tao
    fprintf('>> onlineAs_N(%d) ', i); 
    disp(onlineAs_N(i,:));
end


%% adding new data
minibatchSize = 1;
k = 1;
for t=1:minibatchSize:20
    fprintf('the %dth steps\n', k);
    % get the incoming slice
    endTime = min(tao+t+minibatchSize-1, dims(end));
    idx(end) = {tao+t:endTime};
    
    x = squeeze(X(idx{:}));
    numOfSlice = endTime-tao-t+1;
    % get tensor X of time current time
    idx(end) = {1:endTime};
    Xt = X(idx{:});




    % tensorlab
    tic;
    Ut = cpd(Xt,R,'Algorithm',@cpd_als);
    Ut_N = Ut{end};
    runtime(1, k) = toc;
    normErr(1, k) = frob(Xt - cpdgen(Ut));
    fitness(1, k) = 1-normErr(1, k)/norm(tensor(Xt));

    % % tensor toolbox
    % batchColdOpt.printitn = 0;
    % tic;
    % batchColdXt = cp_als(tensor(Xt), R, batchColdOpt);
    % runtime(2, k) = toc;
    % normErr(2, k) = norm(tensor(Xt)-full(batchColdXt));
    % fitness(2, k) = 1-normErr(2, k)/norm(tensor(Xt));

    % online CP
    tic;
    [onlineAs, onlinePs, onlineQs, onlineAlpha] = onlineCP_update(x, onlineAs, onlinePs, onlineQs);
    onlineAs_N(end+1,:) = onlineAlpha;
    tmp = [onlineAs'; {onlineAs_N}];
    runtime(3, k) = toc;
    normErr(3, k) = norm(tensor(Xt)-full(ktensor(tmp)));
    fitness(3, k) = 1-normErr(3, k)/norm(tensor(Xt));
    for i = tao-5:tao+t
        fprintf('tenlab : (%d)\t', i); 
        disp(Ut_N(i,:));
        fprintf('onl-CP : (%d)\t', i); 
        disp(onlineAs_N(i,:));
    end
    k = k+1;
    input('');
end

% whos onlineAs onlineAs_N;

runtime(end+1:end+2,:) = runtime(1:2,:) - runtime(end,:);
fitness(end+1:end+2,:) = fitness(1:2,:) - fitness(end,:);
normErr(end+1:end+2,:) = normErr(1:2,:) - normErr(end,:);
disp(runtime');
disp(fitness');
disp(normErr');



% onlineAs{end+1} = onlineAs_N;
% T1 = cpdgen(onlineAs);
% whos T1;
% norm(tensor(X)-tensor(T1))
