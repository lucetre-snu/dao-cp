
addpath('tensorlab_2016-03-28');
warning('off','all');

% 375 * 240 * 320 * 3
% 86,400,000

currentPath = fileparts(mfilename('fullpath'));
tensorFile = fopen(strcat(currentPath, '/VideoExtractor/video.tensor'), 'r');

tic;
% Origin Tensor
originX = fscanf(tensorFile, '%d %d %d %d %f', [5, inf]);
valOriginX = (originX(end, :));
subOriginX = (originX(1:4, :))';
dimOriginX = max(subOriginX)
fclose(tensorFile);
toc;

originTensor = struct;
originTensor.val = valOriginX;
originTensor.sub = subOriginX;
originTensor.size = dimOriginX

whos originTensor
toc;

disp('well done');



% R = 4;
% U = rand(dims(1),R);
% V = rand(dims(2),R);
% W = rand(dims(3),R);
% T = cpdgen({U,V,W});
% originT = T;
% T(randperm(numel(T),round(0.1*numel(T)))) = NaN;
% trainT = T;
% T = fmt(T);

% model = struct;
% model.variables.u = randn(dims(1), R);
% model.variables.v = randn(dims(2), R);
% model.variables.w = randn(dims(3), R);
% % model.factors.U = {'u', @struct_nonneg};
% % model.factors.V = {'v', @struct_nonneg};
% % model.factors.W = {'w', @struct_nonneg};
% model.factorizations.ntf.data = T;
% model.factorizations.ntf.cpd = {'U','V','W'};
% sdf_check(model, 'print')
% tic;
% sol = sdf_nls(model, 'Display', 10, 'MaxIter', 100, 'TolX', 1e-5)
% toc;
% U = sol.factors.U;
% V = sol.factors.V;
% W = sol.factors.W;
% indSet = []
% for i = 1:6
%     for j = 1:4
%         x = sub2ind(dims,i,j,1);
%         indSet(end+1) = x;
%     end
% end
% T = cpdgen({U, V, W}, indSet)
% T = cpdgen({U, V, W}, [(1:6)', (7:12)', (13:18)', (19:24)'])
% T = ful({U, V, W},':',':',1)
% T = cpdgen({U, V, W});
% temp = full(T);
% originT(:,:,1)
% trainT(:,:,1)
% temp(:,:,1)