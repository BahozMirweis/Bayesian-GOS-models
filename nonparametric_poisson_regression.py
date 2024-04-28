import numpy as np
from typing import List
from scipy.stats.distributions import invgamma
from tqdm import tqdm
import time
import pickle
from line_profiler import profile
import numba as nb

# r, p_ni params
class Beta_Gos_params:
    def __init__(self, w : np.ndarray):
        n = w.shape[0]
        self._log_r = np.zeros(n + 1)
        self.update_r(w)

    def update_r(self, w: np.ndarray) -> None:
        w_log = np.log(w)
        self._log_r[1:] = np.cumsum(w_log)

    def log_r(self, n : int) -> float:
        return self._log_r[n]
    
    def log_p(self, n: int, i: int) -> float:
        return self._log_r[n] + np.log(np.exp(-self._log_r[i]) - np.exp(-self._log_r[i - 1]))
    
class Gamma_GOS_params:
    def __init__(self, w:np.ndarray):
        n = w.shape[0]
        self._log_r = np.zeros(n + 1)
        self.update_r(w)
        
    def update_r(self, w: np.ndarray) -> None:
        self._log_r[1:] = -np.log(1+np.cumsum(w))

    def log_p(self, n: int, i: int) -> float:
        return self._log_r[n] + np.log(np.exp(-self._log_r[i]) - np.exp(-self._log_r[i - 1]))
    
    def log_r(self, n:int) ->float:
        return self._log_r[n]
    
class Dir_proc_params:
    def __init__(self, alpha) -> None:
        self.alpha = alpha

    def update_r(self, w):
        pass

    def log_p(self, n, i):
        return -np.log(self.alpha + n)
    
    def log_r(self, n):
        return np.log(self.alpha) - np.log(n + self.alpha)

#Gibbs sampling
def gibbs_sampling(y: np.ndarray, X:np.ndarray, iter: int, init_tau_a: float, init_tau_b: float, init_cluster_beta_mu: np.ndarray, init_sigma_beta_a: float, init_sigma_beta_b: float, beta_init:float, alpha_init:float, const_beta_gos:bool, gos_prior:str) -> dict:
    # For beta_gos
    n, p = X.shape
    beta = np.full(n, beta_init)

    if const_beta_gos:
        alpha = np.full(n, alpha_init)
    else:
        alpha = np.arange(start=alpha_init, stop=alpha_init + n, step = 1)

    # Init c_{1:n}
    cluster_graph = {0:0}

    for i in range(1, n):
        cluster_graph[i] = i

    #Init w_{1:n}
    if gos_prior == "gamma":
        w = np.full(n, 1)
        w_params = Gamma_GOS_params(w)
    elif gos_prior == "beta":
        w = sample_posterior_w_beta(cluster_graph, n, alpha, beta)
        w_params = Beta_Gos_params(w)
    else:
        w = np.full(n, 1)
        w_params = Dir_proc_params(1)

    # Init sigma beta
    sigma_beta = 5

    cov_matrix = np.diag(np.full(p, sigma_beta))

    params = {"cluster_values" : [], "W" : np.zeros((iter, n)), "clusters" : []}

    cluster_values = [np.random.multivariate_normal(np.zeros(p), cov_matrix) for _ in range(n)]

    for i in tqdm(range(iter)):
        cluster_values, cluster_graph = sample_non_conj_cluster_posterior(y, X, cluster_graph, cov_matrix, w, w_params, cluster_values)

        # update w
        if gos_prior == "gamma":
            w = sample_posterior_w_gamma(cluster_graph, w, n, alpha, beta, w_params)
        elif gos_prior == "beta":
            w = sample_posterior_w_beta(cluster_graph, n, alpha, beta)

        w_params.update_r(w)

        # update cluster values
        clusters = construct_cluster_list(cluster_graph, n)
        cluster_values = sample_non_conj_cluster_values(clusters, y, X, cov_matrix, cluster_values)

        

        #Add param updates
        params["cluster_values"].append(cluster_values)
        params["W"][i] = w
        params["clusters"].append(cluster_graph)
        cluster_graph = cluster_graph.copy()

    return params


#Posterior w sampling
def sample_posterior_w_beta(cluster_graph : dict[int, int], n: int, alpha:np.ndarray, beta:np.ndarray) -> np.ndarray:
    w = np.zeros(n)

    for i in range(n):
        alpha_count = 0
        beta_count = 0
        for j in range(i + 1, n):
            if cluster_graph[j] < i or cluster_graph[j] == j:
                alpha_count += 1
            
            if cluster_graph[j] == i:
                beta_count += 1
        
        w[i] = np.random.beta(alpha[i] + alpha_count, beta[i] + beta_count)

    return w

def sample_posterior_w_gamma(cluster_graph : dict[int, int], w_prev:np.ndarray, n: int, alpha:np.ndarray, beta:np.ndarray, w_params:Gamma_GOS_params) -> np.ndarray:
    w = w_prev.copy()
    z = np.log(w) 

    iter = 1

    for i in range(n):
        prev_z = z[i]
        prev_log_prob = log_posterior_w_gamma_pdf(z[i], cluster_graph, z, n, i, alpha, beta, w_params)
        for _ in range(iter):
            new_z = prev_z + np.random.normal(0, 0.5)
            new_log_prob = log_posterior_w_gamma_pdf(new_z, cluster_graph, z, n, i, alpha, beta, w_params)
            log_acceptance = min(0, new_log_prob - prev_log_prob)
            log_U = np.log(np.random.uniform())

            if log_U < log_acceptance:
                prev_z = new_z
                prev_log_prob = new_log_prob

        z[i] = prev_z

    return np.exp(z)


def log_posterior_w_gamma_pdf(z : float, cluster_graph : dict[int, int], z_prev:np.ndarray, n: int, i: int, alpha:np.ndarray, beta:np.ndarray, w_params:Gamma_GOS_params) -> float:
    w = np.exp(z_prev)
    w[i] = np.exp(z)

    w_params.update_r(w)

    log_density = alpha[i]*z - beta[i]*np.exp(z) 

    for l in range(i + 1, n):
        if cluster_graph[l] < l:
            log_density += w_params.log_p(l, cluster_graph[l] + 1)
        else:
            log_density += w_params.log_r(l)

    w_params.update_r(np.exp(z_prev))

    return log_density

#Posterior cluster sampling
"""Find log P(c_{index} = k | c_{-index}, w) where k, index start from 1"""
def log_cluster_prior_pred_prob(index:int, k:int, w:np.ndarray, w_params) -> float:
    if k == index:
        return w_params.log_r(index - 1)
    
    return w_params.log_p(index - 1, k)

def construct_cluster_list(graph: dict[int, int], n: int) -> List[List[int]]:
    value_to_cluster = {}
    clusters = []
    for i in range(n):

        if graph[i] == i:
            clusters.append([i])
            value_to_cluster[i] = len(clusters) - 1
        else:
            clusters[value_to_cluster[graph[i]]].append(i)
            value_to_cluster[i] = value_to_cluster[graph[i]]

    return clusters

def construct_cluster_list_without_pos(graph: dict[int, int], pos: int, n: int) -> tuple[List[List[int]], List[int], dict[int, int]]:
    value_to_cluster = {}
    cluster_count = 0
    subtree = [pos]
    for i in range(n):
        if i != pos and graph[i] in subtree:
            subtree.append(i)
        
        if graph[i] == i:
            value_to_cluster[i] = cluster_count
            cluster_count += 1
        else:
            value_to_cluster[i] = value_to_cluster[graph[i]]

    return subtree, value_to_cluster


def sample_non_conj_cluster_posterior(y:np.ndarray, X:np.ndarray,  clusters_graph: dict[int, int], cov_matrix: np.ndarray, w:np.ndarray, w_params, old_cluster_values : List) -> dict[int, int]:
    n, p = X.shape
    ret_graph = clusters_graph.copy()
    cluster_values = old_cluster_values.copy()
    rng = np.random.default_rng()    

    for i in range(n):
        subtree, value_to_cluster = construct_cluster_list_without_pos(ret_graph, i, n)
        log_probs = np.zeros(i + 1)

        for j in range(i + 1):
            log_probs[j] = log_cluster_prior_pred_prob(i + 1, j + 1, w, w_params)

        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        new_val = rng.choice(np.arange(i + 1),p = probs/np.sum(probs))

        if new_val != clusters_graph[i]:
            old_cluster_val = cluster_values[value_to_cluster[i]]

            if new_val == i:
                new_cluster_val = np.random.multivariate_normal(np.zeros(p), cov_matrix)
            else:
                new_cluster_val = cluster_values[value_to_cluster[new_val]]

            log_acc = np.sum(y[subtree]*(X[subtree]@(new_cluster_val - old_cluster_val)) - np.exp(X[subtree]@new_cluster_val) + np.exp(X[subtree]@old_cluster_val))

            log_U = np.log(np.random.uniform())

            if log_U <= log_acc:
                if ret_graph[i] == i:
                    cluster_values.pop(value_to_cluster[i])

                ret_graph[i] = new_val

                if new_val == i:
                    subtree, value_to_cluster = construct_cluster_list_without_pos(ret_graph, -1, n)
                    cluster_values.insert(value_to_cluster[i], new_cluster_val)
        elif new_val == i:
            old_cluster_val = cluster_values[value_to_cluster[i]]
            new_cluster_val = np.random.multivariate_normal(np.zeros(p), cov_matrix)

            log_acc = np.sum(y[subtree]*(X[subtree]@(new_cluster_val - old_cluster_val)) - np.exp(X[subtree]@new_cluster_val) + np.exp(X[subtree]@old_cluster_val))
            log_U = np.log(np.random.uniform())

            if log_U <= log_acc:
                cluster_values[value_to_cluster[i]] = new_cluster_val

    return cluster_values, ret_graph
                
            

def sample_non_conj_cluster_values(clusters : List[List[int]], y:np.ndarray, X:np.ndarray, cov_matrix: np.ndarray, old_cluster_values: List) -> List:
    n, p = X.shape
    num_clusters = len(clusters)
    cluster_values = old_cluster_values.copy()

    for i in range(len(cluster_values)):
        proposal = np.random.multivariate_normal(np.zeros(p), cov_matrix)
        log_accep = 0
        for j in clusters[i]:
            log_accep += y[j]*X[j]@proposal - np.exp(X[j]@proposal) + np.exp(X[j]@cluster_values[i]) - y[j]*X[j]@cluster_values[i]

        log_U = np.log(np.random.uniform())

        if log_U <= log_accep:
            cluster_values[i] = proposal

    return cluster_values


if __name__ == "__main__":    
    iterations = 10000
    data = np.load("poi_data.npz")
    X = data["X"]
    y = data["y"]
    n, p = X.shape

    tests = [ [False, "beta", "long", 1], [True, "beta", "short", 1], [False, "gamma", "long", 3], [True, "gamma", "short", 5], [True, "dir", "short", 1]]

    for test in tests:
        result = gibbs_sampling(y, X, iterations, 2, np.var(y), np.zeros(p), 0.01, 0.01, 1, test[3], test[0], test[1])

        for key, value in result.items():
            print(f"{key} : {value[-1]}")

        last_list = construct_cluster_list(result["clusters"][-1], n)
        print(last_list)

        np.savez(f"poi_reg_results/{test[2]}_mem_{test[1]}_single_params.npz", w = result["W"], X = X, y= y)

        with open(f"poi_reg_results/{test[2]}_mem_{test[1]}_clusters.pkl", "wb") as f:
            pickle.dump(result["clusters"], f)

        with open(f"poi_reg_results/{test[2]}_mem_{test[1]}_cluster_values.pkl", "wb") as f:
            pickle.dump(result["cluster_values"], f, pickle.HIGHEST_PROTOCOL)
    