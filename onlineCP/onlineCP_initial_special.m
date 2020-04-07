function [ Ps, Qs ] = onlineCP_initial_special( initX, As, R )

% if As is not given, calculate the CP decomposition of the initial data
if ~exist('As')
    As = cpd(initX, R, 'Algorithm', @cpd_als);
end

dims = size(initX);
N = length(dims);

% for the first N-1 modes, calculte their assistant matrices P and Q
H = getHadamard(As);
Ks = getKhatriRaoList((As(1:N)));
for n=1:N-1
    Xn = reshape(permute(initX, [n, 1:n-1, n+1:N]), dims(n), []);
    As1 = As{1};
    Ksn = Ks{n};
    whos Xn As1 Ksn
    res = khatrirao(As{1}, Ks{n});
    whos res
    Ps{n} = Xn*khatrirao(As{1}, Ks{n});
    Qs{n} = H./(As{n+1}'*As{n+1});
end
end

