import time
import math
import sys
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.decomposition.candecomp_parafac import initialize_factors, unfolding_dot_khatri_rao, KruskalTensor

class Welford(object):
    def __init__(self,lst=None):
        self.k = 0
        self.M = 0
        self.S = 0
        
        self.__call__(lst)
    
    def update(self,x):
        if x is None:
            return
        self.k += 1
        newM = self.M + (x - self.M)*1./self.k
        newS = self.S + (x - self.M)*(x - newM)
        self.M, self.S = newM, newS

    def consume(self,lst):
        lst = iter(lst)
        for x in lst:
            self.update(x)
    
    def __call__(self,x):
        if hasattr(x,"__iter__"):
            self.consume(x)
        else:
            self.update(x)
            
    @property
    def mean(self):
        return self.M
    @property
    def meanfull(self):
        return self.mean, self.std/math.sqrt(self.k)
    @property
    def std(self):
        if self.k==1:
            return 0
        return math.sqrt(self.S/(self.k-1))
    def __repr__(self):
        return "<Welford: {} +- {}>".format(self.mean, self.std)
    
def construct_tensor(factors):
    weights = tl.ones(factors[0].shape[1])
    est_tensor = tl.kruskal_to_tensor((weights, factors))
    return est_tensor
    
def print_tensor(X, n_digit=1):
    print(np.round(X, n_digit))
    
def compare_tensors(A, B):
    error_norm = tl.norm(A - B)
    print('||A-B||:', error_norm)
    return error_norm
    
def create_tensor_stream(X, start_to_stream, batch_sizes=[]):
    if start_to_stream < 0:
        batch_size = int(-start_to_stream)
        start_to_stream = X.shape[0] % batch_size
        if start_to_stream == 0:
            start_to_stream = batch_size
            batch_sizes = np.full(((X.shape[0]-batch_size) // batch_size), batch_size, dtype=int)
        else:
            batch_sizes = np.full((X.shape[0] // batch_size), batch_size, dtype=int)
        print(start_to_stream, batch_sizes)
    
    total_batch_size = np.sum(batch_sizes)
    if X.shape[0] != start_to_stream + total_batch_size:
        raise ValueError('Total batch size should be the size of streaming part of the tensor.')
    
    X_stream = [X[:start_to_stream]]
    batch_start = start_to_stream
    for batch_size in batch_sizes:
        batch_end = batch_start + batch_size
        X_stream.append(X[batch_start:batch_end])
        batch_start = batch_end
    return np.asarray(X_stream)
    
def get_KhatriRao(factors):
    n_dim = len(factors)
    lefts = [factors[n_dim-1]]
    rights = [factors[0]]
    if n_dim > 2:
        for mode in range(1, n_dim-1):
            lefts.append(tl.tenalg.khatri_rao((lefts[mode-1], factors[n_dim-mode-1])))
            rights.append(tl.tenalg.khatri_rao((factors[mode], rights[mode-1])))
            
    K = lefts.copy()
    K[0] = lefts[n_dim-2]
    K.append(rights[n_dim-2].copy())
    if n_dim > 2:
        for mode in range(1, n_dim-1):
            K[mode] = tl.tenalg.khatri_rao((lefts[n_dim-mode-2], rights[mode-1]))
    return K

def get_KhatriRao_except0(factors):
    n_dim = len(factors)
    lefts = np.empty((n_dim), dtype=object)
    rights = np.empty((n_dim), dtype=object)
    K = np.empty((n_dim), dtype=object)
    
    lefts[1] = factors[n_dim-1]
    rights[1] = factors[1]
    if n_dim > 3:
        for mode in range(2, n_dim-1):
            lefts[mode] = tl.tenalg.khatri_rao((factors[n_dim-mode], lefts[mode-1]))
            rights[mode] = tl.tenalg.khatri_rao((rights[mode-1], factors[mode]))
            
    K[1] = lefts[n_dim-2]
    K[n_dim-1] = rights[n_dim-2]
    if n_dim > 3: 
        for mode in range(2, n_dim-1):
            K[mode] = tl.tenalg.khatri_rao((rights[mode-1], lefts[n_dim-mode-1]))
    return K
    
def get_Hadamard(factors):
    rank = factors[0].shape[1]
    H = tl.tensor(np.ones((rank, rank)))
    for factor in factors:
        H = H * tl.dot(tl.transpose(factor), factor)
    return H


def online_cp(factors_old, X_old, X_new, rank, P, Q, n_iter=1, mu=1, verbose=False, transformed=False):
    mem = 0
    weights = tl.ones(rank)
    if verbose:
        X = tl.tensor(np.concatenate((X_old, X_new)))
    n_dim = tl.ndim(X_old)
    U = factors_old.copy()
    
    if not transformed:
        K = get_KhatriRao_except0(factors_old)
    H = get_Hadamard(factors_old[1:])
    mem += sys.getsizeof(K)
    mem += sys.getsizeof(H)
        
    for i in range(n_iter):
        # temporal mode for A1
        if not transformed:
            mttkrp = tl.dot(tl.unfold(X_new, 0), tl.tenalg.khatri_rao((U[1], K[1])))
        else:
            # for higher accracy, lower speed
            mttkrp_parts = []
            for r in range(rank):
                component = tl.tenalg.multi_mode_dot(X_new, [f[:, r] for f in U], skip=0)
                mttkrp_parts.append(component)
            mttkrp = np.stack(mttkrp_parts, axis=1)
        
        A1 = tl.transpose(tl.solve(tl.transpose(H), tl.transpose(mttkrp)))

        # non-temporal mode
        for mode in range(1, n_dim):
            
            if not transformed:
                dP = tl.dot(tl.unfold(X_new, mode), tl.tenalg.khatri_rao((A1, K[mode])))
                UTU  = tl.dot(tl.transpose(U[mode]), U[mode])
                dQ = tl.dot(tl.transpose(A1), A1) * H / UTU
                mem += sys.getsizeof(dP)
                mem += sys.getsizeof(UTU)
                mem += sys.getsizeof(dQ)
                
                U[mode] = tl.transpose(tl.solve(tl.transpose(mu*Q[mode] + dQ), tl.transpose(mu*P[mode] + dP)))
                P[mode] = P[mode] + dP
                Q[mode] = Q[mode] + dQ
            else:
                U1 = U.copy()
                U1[0] = A1
                
                H_mode  = H / tl.dot(tl.transpose(U[mode]), U[mode])
                V = (mu * tl.dot(tl.transpose(U[0]), U[0]) + tl.dot(tl.transpose(A1), A1)) * H_mode
                
                mttkrp0 = unfolding_dot_khatri_rao(X_old, (None, U), mode)
                mttkrp1 = unfolding_dot_khatri_rao(X_new, (None, U1), mode)
                
                U[mode] = tl.transpose(tl.solve(tl.transpose(V), tl.transpose(mu*mttkrp0 + mttkrp1)))
                H = H_mode * tl.dot(tl.transpose(U[mode]), U[mode])
                
        # temporal mode for A0
        if transformed:
            mttkrp = unfolding_dot_khatri_rao(X_old, (None, U), 0)
            U[0] = tl.transpose(tl.solve(tl.transpose(H), tl.transpose(mttkrp)))
            
        if verbose:
            U1 = U.copy()
            U1[0] = np.concatenate((U[0], A1))
            X_est = construct_tensor(U1)
            compare_tensors(X, X_est)

    U[0] = np.concatenate((U[0], A1))
    return (KruskalTensor((weights, U)), P, Q, mem)


def dtd(factors_old, X_old, X_new, rank, n_iter=1, mu=1, verbose=False):
    mem = 0
    weights = tl.ones(rank)
    if verbose:
        X = tl.tensor(np.concatenate((X_old, X_new)))
    n_dim = tl.ndim(X_old)
    U = factors_old.copy()
    
    for i in range(n_iter):
        # temporal mode for A1
        V = tl.tensor(np.ones((rank, rank)))
        for j, factor in enumerate(U):
            if j != 0:
                V = V * tl.dot(tl.transpose(factor), factor)
        mttkrp = unfolding_dot_khatri_rao(X_new, (None, U), 0)
        mem += sys.getsizeof(mttkrp)
        A1 = tl.transpose(tl.solve(tl.transpose(V), tl.transpose(mttkrp)))

        # non-temporal mode
        for mode in range(1, n_dim):
            U1 = U.copy()
            U1[0] = A1
            V = tl.tensor(np.ones((rank, rank)))
            W = tl.tensor(np.ones((rank, rank)))
            for j, factor in enumerate(U):
                factor_old = factors_old[j]
                if j != mode:
                    W = W * tl.dot(tl.transpose(factor_old), factor)
                    if j == 0:
                        V = V * (mu*tl.dot(tl.transpose(factor), factor) + tl.dot(tl.transpose(A1), A1))
                    else:
                        V = V * tl.dot(tl.transpose(factor), factor)
            mttkrp0 = mu * tl.dot(factors_old[mode], W)
            mttkrp1 = unfolding_dot_khatri_rao(X_new, (None, U1), mode)
            mem += sys.getsizeof(mttkrp0)
            mem += sys.getsizeof(mttkrp1)
            U[mode] = tl.transpose(tl.solve(tl.transpose(V), tl.transpose(mttkrp0 + mttkrp1)))

        # temporal mode for A0
        V = tl.tensor(np.ones((rank, rank)))
        W = tl.tensor(np.ones((rank, rank)))
        for j, factor in enumerate(U):
            factor_old = factors_old[j]
            if j != 0:
                V = V * tl.dot(tl.transpose(factor), factor)
                W = W * tl.dot(tl.transpose(factor_old), factor)
        mttkrp = tl.dot(factors_old[0], W)
        mem += sys.getsizeof(mttkrp)
        U[0] = tl.transpose(tl.solve(tl.transpose(V), tl.transpose(mttkrp)))
        if verbose:
            U1 = U.copy()
            U1[0] = np.concatenate((U[0], A1))
            X_est = construct_tensor(U1)
            compare_tensors(X, X_est)

    U[0] = np.concatenate((U[0].copy(), A1))
    return (KruskalTensor((weights, U)), mem)


def adaptive_online_cp(factors_old, X_old, X_new, rank, n_iter=1, mu=1, verbose=False):
    mem = 0
    weights = tl.ones(rank)
    if verbose:
        X = tl.tensor(np.concatenate((X_old, X_new)))
    n_dim = tl.ndim(X_old)
    U = factors_old.copy()
    
    H = get_Hadamard(U[1:])
    G = H
    
    ATA0 = tl.dot(tl.transpose(U[0]), U[0])
    ATA1 = tl.dot(tl.transpose(U[1]), U[1])
    
    mem += sys.getsizeof(ATA0)
    mem += sys.getsizeof(ATA1)
    mem += sys.getsizeof(G)
    mem += sys.getsizeof(H)
    
    for i in range(n_iter):        
        # temporal mode for A1
        mttkrp_parts = []
        for r in range(rank):
            component = tl.tenalg.multi_mode_dot(X_new, [f[:, r] for f in U], skip=0)
            mttkrp_parts.append(component)
        mttkrp = np.stack(mttkrp_parts, axis=1)
        
        A1 = tl.transpose(tl.solve(tl.transpose(H), tl.transpose(mttkrp)))
        ATA1 = tl.dot(tl.transpose(A1), A1)
        
        
        # non-temporal mode
        for mode in range(1, n_dim):
            
            U1 = U.copy()
            U1[0] = A1
                    
            G = G / tl.dot(tl.transpose(factors_old[mode]), U[mode])
            W = G * tl.dot(tl.transpose(factors_old[0]), U[0])
            mttkrp0 = mu * tl.dot(factors_old[mode], W)
            mttkrp1 = unfolding_dot_khatri_rao(X_new, (None, U1), mode)
            
            H = H / tl.dot(tl.transpose(U[mode]), U[mode])
            V = H * (mu * ATA0 + ATA1)
            U[mode] = tl.transpose(tl.solve(tl.transpose(V), tl.transpose(mttkrp0 + mttkrp1)))
            G = G * tl.dot(tl.transpose(factors_old[mode]), U[mode])
            H = H * tl.dot(tl.transpose(U[mode]), U[mode])
            

        # temporal mode for A0        
        mttkrp = tl.dot(factors_old[0], G)
        U[0] = tl.transpose(tl.solve(tl.transpose(H), tl.transpose(mttkrp)))
        ATA0 = tl.dot(tl.transpose(U[0]), U[0])

        if verbose:
            U1 = U.copy()
            U1[0] = np.concatenate((U[0], A1))
            X_est = construct_tensor(U1)
            compare_tensors(X, X_est)

    U[0] = np.concatenate((U[0].copy(), A1))
    return (KruskalTensor((weights, U)), mem)


from warnings import warn

def get_z_score(x, mean, std):
    if std == 0:
        return 0
    return (x - mean) / std


def online_tensor_decomposition(dataset, X, X_stream, rank, n_iter=1, ul=-1, ll=-1, verbose=False, methods=['dao', 'dtd', 'ocp', 'fcp']):
    results = {}
    start = time.time()
    (weights, factors_old) = parafac(X_stream[0], rank, init='random')
    print('-----------------------------------')
    init_time = time.time()-start
    print('making init decomposition result:', init_time)
    
    for method in methods:

        mem_usage = sys.getsizeof(X_stream[0])
        print('\n >> {} rank-{} n_iter-{}'.format(method, rank, n_iter))
        
        factors = factors_old
        X_old = X_stream[0]
        n_dim = tl.ndim(X_old)
        if not method in ['dao', 'dtd', 'ocp', 'fcp']:
            raise ValueError('The method does not exist.')  
        if method == 'fcp':
            mem_usage = sys.getsizeof(X)
            (weights, factors) = parafac(X, rank, init='random')
            mem_usage += sys.getsizeof(factors)
            X_est = construct_tensor(factors)
            err_norm = tl.norm(X - X_est)
            global_rt = time.time()-start
            global_fit = 1 - (err_norm/tl.norm(X))
            print('global fitness', global_fit)
            print('global running time', global_rt)
            print('memory usage', mem_usage)
            results[method] = [global_fit, 0, global_rt, 0, mem_usage, X_est]
            continue

        ktensors = []
        verbose_list = []
        split_points = []
        refine_points = []
        fitness = []
        running_time = []
        begin = time.time() - init_time

        welford = Welford()
        X_est = construct_tensor(factors)
        err_norm = tl.norm(X_old - X_est)
        welford(err_norm * 1.2)


        if method == 'ocp':
            start = time.time()
            K = get_KhatriRao_except0(factors)
            H = get_Hadamard(factors)

            P = np.empty((n_dim), dtype=object)
            Q = np.empty((n_dim), dtype=object)

            for mode in range(1, n_dim):
                P[mode] = tl.dot(tl.unfold(X_old, mode), tl.tenalg.khatri_rao((factors[0], K[mode])))
                Q[mode] = H / tl.dot(tl.transpose(factors[mode]), factors[mode])
            #print('init_time:', time.time()-start)
            mem_usage += sys.getsizeof(K)
            mem_usage += sys.getsizeof(H)
            mem_usage += sys.getsizeof(P)
            mem_usage += sys.getsizeof(Q)
        
        iter_mem_usage = 0
        for i, X_new in enumerate(X_stream[1:]):
            i_mem = sys.getsizeof(X_new)
            start = time.time()
            if method == 'dao':
                ((weights, factors0), mem) = adaptive_online_cp(factors.copy(), X_old, X_new, rank, n_iter=n_iter, mu=0.8, verbose=False)
            elif method == 'ocp':
                ((weights, factors0), P0, Q0, mem) = online_cp(factors.copy(), X_old, X_new, rank, P, Q, verbose=False)
            elif method == 'dtd':
                ((weights, factors0), mem) = dtd(factors.copy(), X_old, X_new, rank, verbose=False)
            
            i_mem += mem
            U = factors0.copy()
            i_mem += sys.getsizeof(U)
            U[0] = U[0][-X_new.shape[0]-1:-1]
            dX_est = construct_tensor(U)

            err_norm = tl.norm(X_new - dX_est)
            z_score = get_z_score(err_norm, welford.mean, welford.std)

            if method == 'dao' and ul > 0 and z_score > ul:
                weights = tl.ones(rank)
                ktensors.append(KruskalTensor((weights, factors.copy())))
                #print('=== SPLIT({}, {}) ==='.format(z_score, err_norm))
                split_points.append(i+1)

                X_old = X_stream[i+1]

                (weights, factors0) = parafac(X_old, rank, init='random')
                elapsed_time = time.time()-start
                #print('making init decomposition result:', time.time()-start)
                verbose_list.append([i+1, elapsed_time, err_norm, z_score])

                i_mem += sys.getsizeof(factors0)
                start = time.time()
                X_est = construct_tensor(factors0)
                err_norm = tl.norm(X_old - X_est)
                welford = Welford()
                welford(err_norm * 1.2)

                z_score = get_z_score(err_norm, welford.mean, welford.std)
                factors = factors0.copy()
                welford(err_norm)
                elapsed_time = time.time()-start
                #print('{}th_iter:'.format(i+1), elapsed_time, err_norm, z_score)
                verbose_list.append([i+1, elapsed_time, err_norm, z_score])
                fitness.append(err_norm/tl.norm(X_new))
                running_time.append(elapsed_time)
                continue
            elif method == 'dao' and ll > 0 and z_score > ll:
                #print('=== REFINE({}, {}) ==='.format(z_score, err_norm))
                refine_points.append(i+1)
                elapsed_time = time.time()-start
                verbose_list.append([i+1, elapsed_time, err_norm, z_score])

                ((weights, factors), mem) = adaptive_online_cp(factors, X_old, X_new, rank, n_iter=n_iter*2, mu=0.5, verbose=False)
                
                i_mem += mem
                i_mem += sys.getsizeof(factors)
                U = factors.copy()
                U[0] = U[0][-X_new.shape[0]-1:-1]
                dX_est = construct_tensor(U)
                err_norm = tl.norm(X_new - dX_est)
                welford(err_norm)
            else:
                if method == 'ocp':
                    P = P0
                    Q = Q0
                factors = factors0.copy()
                welford(err_norm)
            
            elapsed_time = time.time()-start
            #print('{}th_iter:'.format(i+1), elapsed_time, err_norm, z_score)
            verbose_list.append([i+1, elapsed_time, err_norm, z_score])
            fitness.append(err_norm/tl.norm(X_new))
            running_time.append(elapsed_time)
            X_old = tl.concatenate((X_old, X_new))
            iter_mem_usage = max(iter_mem_usage, i_mem)
            if verbose:
                X_est = construct_tensor(factors)
                compare_tensors(X_old, X_est)

        mem_usage += iter_mem_usage
        
        weights = tl.ones(rank)
        ktensors.append(KruskalTensor((weights, factors)))
        mem_usage += sys.getsizeof(ktensors)
        
#     return (ktensors, np.asarray(verbose_list))
        global_rt = time.time() - begin

        tensor_est = construct_tensor(ktensors[0][1])
        for (weights, factors) in ktensors[1:]:
            tensor_est = tl.tensor(tl.concatenate((tensor_est, construct_tensor(factors))))
        global_error_norm = compare_tensors(X, tensor_est)
#        print('Elapsed Time:', time.time() - begin)
#         print_tensor(np.asarray((X, tensor_est))[:,0,0,0,:10])
        if method == 'dao':
            print('split:', len(split_points), 'refine:', len(refine_points))
            
        if method != 'fcp':
            verbose_list = np.asarray(verbose_list, dtype=float)
            fitness = np.asarray(fitness, dtype=float)
            running_time = np.asarray(running_time, dtype=float)

            tot_norm = tl.norm(X)
            local_fit = 1 - np.mean(fitness)
            local_rt = np.mean(running_time)
            global_fit = 1 - (global_error_norm / tot_norm)
            print('global fitness', global_fit)
            print('local fitness', local_fit)
            print('global running time', global_rt)
            print('local running time', local_rt)
            print('memory usage', mem_usage)
            results[method] = [global_fit, local_fit, global_rt, local_rt, mem_usage, verbose_list, (split_points, refine_points), tensor_est]
            
    return results