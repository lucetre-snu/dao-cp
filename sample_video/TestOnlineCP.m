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
dims = [180 320 3 205];
% dims = squeeze(permute(dims, [2 3 4 1]));
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
            videoTensor(row(2)-30, row(3), row(4), row(1)) = row(5);
        end
    end
    fclose(tensorFile);
    toc;
end

tic;
tao = 5;
idx = repmat({':'}, 1, length(dims));
idx(end) = {1:tao};
initX = videoTensor(:, :, :, 1:tao);

options.Display = true; % Show progress on the command line.
options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
options.Refinement = false;
initAs = cpd(initX, R, options);
% As1 = initAs{1};
% As2 = initAs{2};
% As3 = initAs{3};
% As4 = initAs{4}
% whos initX As1 As2 As3 As4
toc;

[onlinePs, onlineQs] = onlineCP_initial_tenlab(initX, initAs, R);
onlineAs = initAs(1:end-1);
onlineAs_N = initAs{end};
whos onlinePs onlineQs onlineAs_N
toc;

minibatchSize = 1;
k = 1;
for t = 1:minibatchSize:10
    fprintf('the %dth steps\n', t);
    % get the incoming slice
    endTime = min(tao+t+minibatchSize-1, dims(end));
    idx(end) = {tao+t:endTime};
    
    x = squeeze(videoTensor(idx{:}));
    numOfSlice = endTime-tao-t+1;
    idx(end) = {1:endTime};
    Xt = videoTensor(idx{:});
    whos Xt
    tic;
    [onlineAs, onlinePs, onlineQs, onlineAlpha] = onlineCP_update(x, onlineAs, onlinePs, onlineQs);
    onlineAs_N(end+1,:) = onlineAlpha;
    Uest = [onlineAs'; {onlineAs_N}];
    Test = cpdgen(Uest);
    [frob(Test), frob(Xt), frob(Test-Xt)] 
    Test = permute(cpdgen(Uest), [4 1 2 3]);

    imwrite(uint8(squeeze(Test(t, :, :, :))), strcat('./video_frame/video_frame', num2str(t), '_OnlineCP', num2str(R), '.jpg'));

    % runtime(3, k) = toc;
    % normErr(3, k) = norm(tensor(Xt)-full(ktensor(tmp)));
    % fitness(3, k) = 1-normErr(3, k)/norm(tensor(Xt));
    % for i = tao-5:tao+t
    %     fprintf('tenlab : (%d)\t', i); 
    %     disp(Ut_N(i,:));
    %     fprintf('onl-CP : (%d)\t', i); 
    %     disp(onlineAs_N(i,:));
    % end
    % k = k+1;
    % input('');

end

% for frame = 1:numOfFrames
%     disp(frame);
%     T = videoTensor(1:frame, :, :, :);
%     whos T
%     options.Display = true; % Show progress on the command line.
%     options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
%     options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
%     options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
%     options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
%     options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
%     options.Refinement = false;
%     options.incomplete = false;
%     Uest = cpd(T, R, options);
%     whos Uest
%     Test = cpdgen(Uest);

%     % imwrite(uint8(squeeze(T(frame, :, :, :))), strcat('./video_frame/video_frame', num2str(frame), '_org.jpg'));
%     imwrite(uint8(squeeze(Test(frame, :, :, :))), strcat('./video_frame/video_frame', num2str(frame), '_est', num2str(R), '.jpg'));
% end