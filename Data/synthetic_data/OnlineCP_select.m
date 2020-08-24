addpath('../packages/tensorlab_2016-03-28');
addpath('../packages/tensor_toolbox-v3.1');
addpath('../packages/onlineCP');
warning('off', 'all');

% OnlineCP w. select in synthetic data


% method="OnlineCP_select"; OnlineCP_select
% method="OnlineCP_trigger"; OnlineCP_select
% method="OnlineCP_split"; OnlineCP_select
% method="OnlineCP"; OnlineCP_select

currentPath = fileparts(mfilename('fullpath'));
options.Display = false; % Show progress on the command line.
options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
options.Refinement = false;

startFrame = 0;
endFrame = 1000;
numOfFrames = endFrame - startFrame;

size_tens = [numOfFrames 10 20 30];
dims = [10 20 30 numOfFrames];
R = 10;

if method == "OnlineCP_select"
    update_threshold = 20;
    split_threshold = 500;
elseif method == "OnlineCP_trigger"
    update_threshold = 20;
    split_threshold = 9999;
elseif method == "OnlineCP_split"
    update_threshold = 9999;
    split_threshold = 500;
elseif method == "OnlineCP"
    update_threshold = 9999;
    split_threshold = 9999;
end


tao = 5;
iterFrame = 5;
T = NaN(dims);
errNormHistory = strings([1 numOfFrames]);
minibatchSize = 1;

tic;
for i = startFrame/iterFrame:endFrame/iterFrame-1
    tensorFile = fopen(strcat(currentPath, '/data_normal/data', num2str(i), '.tensor'), 'r');
    % tensorFile = fopen(strcat(currentPath, '/data_factor/data', num2str(i), '.tensor'), 'r');
    X = fscanf(tensorFile, "%d %d %d %d %f", [5, inf]);
    for row = X
        T(row(2), row(3), row(4), row(1)-startFrame) = row(5);
    end
    fclose(tensorFile);
end
toc;

whos T


idx = repmat({':'}, 1, length(dims));
prevDrasticFrame = 1;

for frame = 1:minibatchSize:numOfFrames
    t = frame;
    % fprintf('\n> %dth frame\n', frame+startFrame);
    endTime = min(frame+minibatchSize-1, numOfFrames);
    idx(end) = {frame:endTime};
    x = squeeze(T(idx{:}));

    idx(end) = {prevDrasticFrame:endTime};
    Xt = T(idx{:});
    imgOrg = squeeze(Xt(:, :, :, frame-prevDrasticFrame+1));

    tic;

    if frame-prevDrasticFrame < tao-1
        % disp('continue.');
        testFrame(t) = frame+startFrame;
        testRuntime(t) = toc;
        continue;

    elseif frame-prevDrasticFrame+1 == tao
        % disp('OnlineCP init! CP-ALS update!');
        initAs = cpd(Xt, R, options);
    
        [onlinePs, onlineQs] = onlineCP_initial_tenlab(Xt, initAs, R);
        onlineAs = initAs(1:end-1);
        onlineAs_N = initAs{end};
    
        Uest = [onlineAs'; {onlineAs_N}];
        Test = cpdgen(Uest);

        testFrame(t) = frame+startFrame;
        testRuntime(t) = toc;

        prevImgNormErr = 0;
        maxErrNormTol = 0;
        for i = 1:tao
            prevFrame = frame-prevDrasticFrame+1-tao+i;
            imgEst = squeeze(Test(:, :, :, prevFrame));
            imgOrg = squeeze(Xt(:, :, :, prevFrame));
            testImgNormErr(t-tao+i) = frob(imgEst-imgOrg);
            errNormHistory(t-tao+i) = sprintf("%.f\t%s", frob(imgEst-imgOrg), errNormHistory(t-tao+i));

            % if t > 1
            %     errNormTol = testImgNormErr(t)-testImgNormErr(t-1);
            %     if maxErrNormTol < errNormTol
            %         maxErrNormTol = errNormTol;
            %     end
            % end

            if prevImgNormErr < testImgNormErr(t-tao+i)
                prevImgNormErr = testImgNormErr(t-tao+i);
            end
        end


        continue;

    else
        % disp('OnlineCP update!');
        [onlineAs, onlinePs, onlineQs, onlineAlpha] = onlineCP_update(x, onlineAs, onlinePs, onlineQs);
        onlineAs_N(end+1,:) = onlineAlpha;
        Uest = [onlineAs'; {onlineAs_N}];
    
        As1 = Uest{1};
        As2 = Uest{2};
        As3 = Uest{3};
        As4 = Uest{4};
    
        Test = cpdgen(Uest);
        imgEst = squeeze(Test(:, :, :, frame-prevDrasticFrame+1));
        testImgNormErr(t) = frob(imgEst-imgOrg);
        errNormHistory(t) = sprintf("%.f\t%s", frob(imgEst-imgOrg), errNormHistory(t));
        
        errNormTol = testImgNormErr(t)-testImgNormErr(t-1);
        if split_threshold < errNormTol
            disp(strcat(num2str(frame), 'OnlineCP-split triggered! Drastic scene detected.'));

            errNormTol / maxErrNormTol
            prevDrasticFrame = frame;
            testFrame(t) = frame+startFrame;
            testRuntime(t) = toc;
            prevImgNormErr = 0;

        elseif update_threshold < errNormTol
            disp(strcat(num2str(frame), 'OnlineCP-update triggered! Similar scene detected.'));
            disp(prevDrasticFrame);


            disp('OnlineCP init! CP-ALS update!');
            initAs = cpd(Xt, R, options);
        
            [onlinePs, onlineQs] = onlineCP_initial_tenlab(Xt, initAs, R);
            onlineAs = initAs(1:end-1);
            onlineAs_N = initAs{end};
        
            Uest = [onlineAs'; {onlineAs_N}];
            Test = cpdgen(Uest);
    
            testFrame(t) = frame+startFrame;
            testRuntime(t) = toc;
    
            prevImgNormErr = 0;
            for i = prevDrasticFrame:frame
                prevFrame = i-prevDrasticFrame+1;

                imgEst = squeeze(Test(:, :, :, prevFrame));
                imgOrg = squeeze(Xt(:, :, :, prevFrame));

              
                testImgNormErr(i) = frob(imgEst-imgOrg);
                errNormHistory(i) = sprintf("%.f\t%s", frob(imgEst-imgOrg), errNormHistory(i));
                if prevImgNormErr < testImgNormErr(i)
                    prevImgNormErr = testImgNormErr(i);
                end
            end

            continue;
        end
    end
    % toc;

    testFrame(t) = frame+startFrame;
    testRuntime(t) = toc;
    testImgNormErr(t) = frob(imgEst-imgOrg);
    % errNormHistory(t) = sprintf("%.f\t%s", frob(imgEst-imgOrg), errNormHistory(t));
    if prevImgNormErr < testImgNormErr(t)
        prevImgNormErr = testImgNormErr(t);
    end
    % if maxErrNormTol < errNormTol
    %     maxErrNormTol = errNormTol;
    % end
end
% whos As1 As2 As3 As4
filename = strcat('result/', method, '.txt');
fileID = fopen(filename, 'w');
whos testRuntime testImgNormErr errNormHistory

for frame = 1:minibatchSize:numOfFrames
    fprintf(fileID, '%d\t%.4f\t%s\n', frame, testRuntime(frame), errNormHistory(frame));
end
fclose(fileID);
