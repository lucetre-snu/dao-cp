addpath('../tensorlab_2016-03-28');
addpath('../tensor_toolbox-v3.1');
addpath('../onlineCP');
warning('off', 'all');


% 205 * 180(31~210) * 320 * 3

currentPath = fileparts(mfilename('fullpath'));
videoTensor = [];
R = 10;
dims = [205 180 320 3];
numOfFrames = dims(1);
iterFrame = 5;
% N = numOfFrames / iterFrame;
N = 2;
videoTensor = NaN(dims);

for i = 1:N
    tensorFile = fopen(strcat(currentPath, '/video', num2str(i-1), '.tensor'), 'r');
    disp(i);
    tic;
    X = fscanf(tensorFile, "%d %d %d %d %d", [5, inf]);
    for row = X
        if row(2) > 30 & row(2) <= 210
            videoTensor(row(1), row(2)-30, row(3), row(4)) = row(5);
        end
    end
    fclose(tensorFile);
    toc;
end


tic;
initX = videoTensor(1:5, :, :, :);

options.Display = true; % Show progress on the command line.
options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
options.Refinement = false;
initAs = cpd(initX, R, options);

whos initX initAs
toc;

% initialize onlineCP method
[onlinePs, onlineQs] = onlineCP_initial_tenlab(initX, initAs, R);
onlineAs = initAs(1:end-1);
onlineAs_N = initAs{end};
for i = 1:3
    fprintf('>> onlineAs_N(%d) ', i); 
    disp(onlineAs_N(i,:));
end

toc;


% minibatchSize = 1;
% k = 1;
% for frame = 1:numOfFrames
%     fprintf('the %dth steps\n', k);
%     % get the incoming slice
%     endTime = min(tao+t+minibatchSize-1, dims(end));
%     idx(end) = {tao+t:endTime};
    
%     x = squeeze(X(idx{:}));
%     numOfSlice = endTime-tao-t+1;
%     % get tensor X of time current time
%     idx(end) = {1:endTime};
%     Xt = X(idx{:});

for frame = 1:numOfFrames
    T = videoTensor(1:frame, :, :, :);
    whos T
    options.Display = true; % Show progress on the command line.
    options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
    options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
    options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
    options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
    options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
    options.Refinement = false;
    options.incomplete = false;
    Uest = cpd(T, R, options);
    whos Uest
    Test = cpdgen(Uest);

    % imwrite(uint8(squeeze(T(frame, :, :, :))), strcat('./video_frame/video_frame', num2str(frame), '_org.jpg'));
    imwrite(uint8(squeeze(Test(frame, :, :, :))), strcat('./video_frame/video_frame', num2str(frame), '_est', num2str(R), '.jpg'));
end