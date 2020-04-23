addpath('../tensorlab_2016-03-28');
addpath('../tensor_toolbox-v3.1');
addpath('../onlineCP');
warning('off', 'all');

% 205 * 180(31~210) * 320 * 3

currentPath = fileparts(mfilename('fullpath'));
videoTensor = [];
R = 10;
dims = [205 180 320 3];
% numOfFrames = dims(1);
% tao = 100;
numOfFrames = 10;
tao = 5;
dims = [180 320 3 205];
iterFrame = 5;
N = numOfFrames / iterFrame;
videoTensor = NaN(dims);

outputVideo = VideoWriter('test');
outputVideo.FrameRate = 30 * 0.25;
open(outputVideo);

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

% tic;
idx = repmat({':'}, 1, length(dims));
idx(end) = {1:tao};
initX = videoTensor(:, :, :, 1:tao);


options.Display = false; % Show progress on the command line.
options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
options.Refinement = false;
initAs = cpd(initX, R, options);

[onlinePs, onlineQs] = onlineCP_initial_tenlab(initX, initAs, R);
onlineAs = initAs(1:end-1);
onlineAs_N = initAs{end};
whos onlineAs_N
toc;


minibatchSize = 1;
for t = 1:minibatchSize:numOfFrames-tao
    fprintf('the %dth steps\n', t);
    % get the incoming slice
    frame = tao+t;
    endTime = min(tao+t+minibatchSize-1, numOfFrames);
    idx(end) = {tao+t:endTime};
    
    x = squeeze(videoTensor(idx{:}));
    idx(end) = {1:endTime};
    Xt = videoTensor(idx{:});
    tic;
    [onlineAs, onlinePs, onlineQs, onlineAlpha] = onlineCP_update(x, onlineAs, onlinePs, onlineQs);
    onlineAs_N(end+1,:) = onlineAlpha;
    Uest = [onlineAs'; {onlineAs_N}];

    As1 = Uest{1};
    As2 = Uest{2};
    As3 = Uest{3};
    As4 = Uest{4};
    [As1(1:3,1), As2(1:3,1), As3(1:3,1)]

    Test = cpdgen(Uest);
    testRuntime(t) = toc;
    testNormErr(t) = frob(Test-Xt);
    testFitness(t) = (1-testNormErr(t)/frob(Xt))*100;
    toc;
    Test = permute(cpdgen(Uest), [4 1 2 3]);
    img = uint8(squeeze(Test(frame, :, :, :)));
    writeVideo(outputVideo,img);
end
close(outputVideo);

whos As1 As2 As3 As4
[Uest{1}(:,1)', Uest{2}(:,1)' Uest{3}(:,1)', Uest{4}(:,1)'];
disp(onlineAs_N(:,1:3))

testRuntime_Fitness = [testRuntime', testFitness']
