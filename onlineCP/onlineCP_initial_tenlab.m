function [ Ps, Qs ] = onlineCP_initial_tenlab( initX, As, R )

% if As is not given, calculate the CP decomposition of the initial data
if ~exist('As')
    As = cpd(initX, R, 'Algorithm', @cpd_als);
end

dims = size(initX);
N = length(dims);

% for the first N-1 modes, calculte their assistant matrices P and Q
H = getHadamard(As);
Ks = getKhatriRaoList((As(1:N-1)));
for n=1:N-1
    Xn = reshape(permute(initX, [n, 1:n-1, n+1:N]), dims(n), []);
    Ps{n} = Xn*khatrirao(As{N}, Ks{n});
    Qs{n} = H./(As{n}'*As{n});
end
end

