%% Shuo Zhou, Xuan Vinh Nguyen, James Bailey, Yunzhe Jia, Ian Davidson,
% "Accelerating Online CP Decompositions for Higher Order Tensors",
% (C) 2016 Shuo Zhou   
% Email: zhous@student.unimelb.edu.au

% To run the code, Tensor Toolbox is required.
% Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor Toolbox 
% Version 2.6, Available online, February 2015. 
% URL: http://www.sandia.gov/~tgkolda/TensorToolbox/

%% This is a demo to compare OnlineCP with classic ALS algorithms
% Batch Cold is typical ALS implemented in Tensor Toolbox
% Batch Hot uses the previous result as the initialization to ALS

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

size_tens = [10 20 30];
T = randn(size_tens);
whos T;


disp('>> X loading');
load X.mat;
whos X

Uest = cpd(X,R,'Algorithm',@cpd_als);
Uest_N = Uest{end};
for i = tao-5:tao
    fprintf('(%d)\t', i); 
    disp(Uest_N(i,:));
end

%% initialization
% get initX
idx = repmat({':'}, 1, length(dims));
idx(end) = {1:tao};
initX = X(idx{:});

% factorize initX 
initOpt.printitn = 1;
initOpt.maxiters = 100;
initOpt.tol = 1e-8;
estInitX = cp_als(tensor(initX), R, initOpt);
initAs = estInitX.U;
% absorb lambda into the last dimension
initAs{end} = initAs{end}*diag(estInitX.lambda);

% initilize each algorithm
% batch 
batchHotAs = initAs;
% initialize onlineCP method
[onlinePs, onlineQs] = onlineCP_initial(initX, initAs, R);
onlineAs = initAs(1:end-1);
onlineAs_N = initAs{end};
for i = tao-5:tao
    fprintf('(%d)\t', i); 
    disp(onlineAs_N(i,:));
end

%% adding new data
minibatchSize = 1;
k = 1;
TT = 5;
for t=1:minibatchSize:TT
    fprintf('the %dth steps\n', k);
    % get the incoming slice
    endTime = min(tao+t+minibatchSize-1, dims(end));
    idx(end) = {tao+t:endTime};
    
    x = squeeze(X(idx{:}));
    numOfSlice = endTime-tao-t+1;
    % get tensor X of time current time
    idx(end) = {1:endTime};
    Xt = X(idx{:});


    Ut = cpd(Xt,R,'Algorithm',@cpd_als);
    Ut_N = Uest{end};

    % cp-als using tensor box
    batchColdOpt.printitn = 0;
    tic;
    batchColdXt = cp_als(tensor(Xt), R, batchColdOpt);
    batchColdAs = batchColdXt.U;
    % absorb lambda into the last dimension
    batchColdAs{end} = batchColdAs{end}*diag(batchColdXt.lambda);
    batchColdAs_N = batchColdAs{end};
    runtime = toc;
    time(1, k) = runtime;

    % cp-als using tensor box
    batchColdOpt.printitn = 0;
    tic;
    batchColdXt = cp_als(tensor(Xt), R, batchColdOpt);
    batchColdAs = batchColdXt.U;
    % absorb lambda into the last dimension
    batchColdAs{end} = batchColdAs{end}*diag(batchColdXt.lambda);
    batchColdAs_N = batchColdAs{end};
    runtime = toc;
    time(1, k) = runtime;
    % fitness(1, k) = 1-(norm(tensor(Xt)-full(batchColdXt))/norm(tensor(Xt)));
%     (norm(tensor(Xt)-full(batchColdXt))/norm(tensor(Xt)))
    % diff_cold = norm(tensor(Xt)-full(batchColdXt))
%     % hot batch
%     tic;
%     % estimate the projection of new data on the time mode
%     batchHotAlpha = reshape(permute(x, [N, 1:N-1]), numOfSlice, [])...
%         *khatrirao(batchHotAs(1:end-1), 'r')/getHadamard(batchHotAs(1:end-1));
%     batchHotAs{end} = [batchHotAs{end}; batchHotAlpha];
%     batchHotOpt.printitn = 0;
%     batchHotOpt.init = batchHotAs;
%     batchHotXt = cp_als(tensor(Xt), R, batchHotOpt);
%     batchHotAs = batchHotXt.U;
%     batchHotAs{end} = batchHotAs{end}*diag(batchHotXt.lambda);
%     runtime = toc;
%     time(2, k) = runtime;
%     fitness(2, k) = 1-(norm(tensor(Xt)-full(batchHotXt))/norm(tensor(Xt)));

    % online CP
    tic;
    [onlineAs, onlinePs, onlineQs, onlineAlpha] = onlineCP_update(x, onlineAs, onlinePs, onlineQs);
    onlineAs_N(end+1,:) = onlineAlpha;
    runtime = toc;
    tmp = [onlineAs; {onlineAs_N}];
    for i = tao-5:tao+t
        fprintf('tenlab : (%d)\t', i); 
        disp(Ut_N(i,:));
        % fprintf('tenbox : (%d)\t', i); 
        % disp(batchColdAs_N(i,:));
        % fprintf('onl-CP : (%d)\t', i); 
        % disp(onlineAs_N(i,:));
    end
    time(3, k) = runtime;
    % fitness(3, k) = 1-(norm(tensor(Xt)-full(ktensor(tmp)))/norm(tensor(Xt)));
    % diff_onlineCP = (norm(tensor(Xt)-full(ktensor(tmp))))
    % disp(diff_onlineCP - diff_cold);

    input('');
    k = k+1;
end

% %% plot results
% ts = [1:TT];
% gap = floor(TT/20)-1;
% 
% % CPU time vs steps
% figure
% semilogy(ts(1:gap:end),time(1,1:gap:end),'-kv');hold on;
% semilogy(ts(1:gap:end),time(2,1:gap:end),'-k^');
% semilogy(ts(1:gap:end),time(3,1:gap:end),'-k','LineWidth',2);
% xlabel('Steps');
% ylabel('Runing Time');
% legend('Batch Cold', 'Batch Hot', 'OnlineCP');
% grid on;
% set(gca,'FontSize',14)
% 
% % Fitness vs steps
% figure
% plot(ts(1:gap:end),fitness(1,1:gap:end),'-kv');hold on;
% plot(ts(1:gap:end),fitness(2,1:gap:end),'-k^');
% plot(ts(1:gap:end),fitness(3,1:gap:end),'-k','LineWidth',2);
% xlabel('Steps');
% ylabel('Fitness');
% legend('Batch Cold', 'Batch Hot', 'OnlineCP');
% grid on;
% set(gca,'FontSize',14)