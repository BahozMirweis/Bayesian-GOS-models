import numpy as np
from typing import List
from scipy.stats.distributions import invgamma
from scipy.linalg import toeplitz
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
def gibbs_sampling(y: np.ndarray, X:np.ndarray, iter: int, init_tau_a: float, init_tau_b: float, init_cluster_beta_mu: np.ndarray, init_sigma_beta_a: float, init_sigma_beta_b: float, beta_init:float, alpha_init:float, const_beta_gos:bool, gos_prior:str, ar_cov:bool) -> dict:
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
        #w = sample_posterior_w_beta(cluster_graph, n, alpha, beta)
        w = np.full(n, 0.5)
        w_params = Beta_Gos_params(w)
    else:
        w = np.full(n, 1)
        w_params = Dir_proc_params(1)

    # Init tau
    tau = [invgamma.rvs(init_tau_a, scale = init_tau_b) for _ in range(n)]

    # Init sigma beta

    rho = 0.5
    sigma = 1

    #sigma_beta = 5

    params = {"cluster_values" : [], "W" : np.zeros((iter, n)), "tau" : [], "rho": np.zeros(iter), "clusters" : []}

    cluster_values = [np.random.multivariate_normal(np.zeros(p), 0.5*np.identity(p)) for i in range(n)]
    

    for i in tqdm(range(iter)):
        # update clusters        
        if ar_cov:
            row = [rho**i for i in range(p)]
            cov_matrix = sigma*toeplitz(row, row)
            Pinv = create_Lambda(rho, sigma, p)
        else:
            cov_matrix = np.identity(p)*0.5

        cluster_values, tau, cluster_graph =  sample_non_conj_cluster_posterior(y, X, cluster_graph, tau, w, w_params, cluster_values, init_tau_a, init_tau_b, cov_matrix)

        # Update w
        if gos_prior == "gamma":
            w = sample_posterior_w_gamma(cluster_graph, w, n, alpha, beta, w_params)
        elif gos_prior == "beta":
            w = sample_posterior_w_beta(cluster_graph, n, alpha, beta)

        w_params.update_r(w)

        # update cluster values
        clusters = construct_cluster_list(cluster_graph, n)

        if ar_cov:
            rho = sample_posterior_rho(rho, sigma, tau, len(clusters), p, cluster_values, Pinv)

        #Add param updates
        params["cluster_values"].append(cluster_values)
        params["W"][i] = w
        params["tau"].append(tau)
        params["rho"][i] = rho
        params["clusters"].append(cluster_graph)
        cluster_graph = cluster_graph.copy()

    return params

#fast linalg methods
@nb.njit(fastmath=True)
def create_Lambda(rho, sigma,p):
    diagonal = np.full(p, 1 + rho**2)
    diagonal[0] = 1
    diagonal[-1] = 1
    diag_above = np.full(p - 1, -rho)

    return 1/(1-rho**2)*(np.diag(diagonal) + np.diag(diag_above, 1) + np.diag(diag_above, -1))


#update rho and sigma
def rho_sigma_exp(cluster_values, Pinv):
    total = 0
    for i in range(len(cluster_values)):
        total += cluster_values[i]@Pinv@cluster_values[i].T

    return total

def sample_posterior_sigma(num_clusters, p, Pinv, cluster_values):
    return invgamma.rvs(num_clusters*p/2, scale = rho_sigma_exp(cluster_values, Pinv)/2)

def sample_posterior_rho(rho, sigma, tau, num_clusters, p, cluster_values, Pinv_init):
    w = np.log(rho/(1 - rho))
    w += np.random.normal(0, 0.5)
    
    new_rho = np.exp(w)/(1 + np.exp(w))
    newPinv = create_Lambda(new_rho, sigma, p)

    old_pdf = log_rho_pdf(rho, sigma, num_clusters, p, cluster_values, Pinv_init)
    new_pdf = log_rho_pdf(new_rho, sigma, num_clusters, p, cluster_values, newPinv)
    
    log_acceptance = min(0, new_pdf - old_pdf)
    log_unif = np.log(np.random.uniform())

    if log_unif < log_acceptance:
        return new_rho

    return rho

def log_rho_pdf(rho, sigma, num_clusters, p, cluster_values, Pinv):
    return np.log(1 + rho)/2 + (p - 1)*num_clusters/2*np.log(1 - rho**2) - 1/(2*sigma)*rho_sigma_exp(cluster_values, Pinv) + np.log(rho*(1-rho))


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
    clusters = []
    subtree = [pos]
    for i in range(n):
        if i == pos:
            continue
        elif graph[i] in subtree:
            subtree.append(i)
        elif graph[i] == i:
            clusters.append([i])
            value_to_cluster[i] = len(clusters) - 1
        else:
            clusters[value_to_cluster[graph[i]]].append(i)
            value_to_cluster[i] = value_to_cluster[graph[i]]

    return clusters, subtree, value_to_cluster

def construct_subtree(graph: dict[int, int], pos: int, n: int) -> tuple[List[List[int]], List[int], dict[int, int]]:
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

def sample_non_conj_cluster_posterior(y:np.ndarray, X:np.ndarray,  clusters_graph: dict[int, int], old_tau_sq, w:np.ndarray, w_params, old_cluster_values, tau_a, tau_b, cov_matrix):
    n, p = X.shape
    ret_graph = clusters_graph.copy()
    cluster_values = old_cluster_values.copy()
    tau_sq = old_tau_sq.copy()
    rng = np.random.default_rng()    

    for i in range(n):
        subtree, value_to_cluster = construct_subtree(ret_graph, i, n)
        log_probs = np.zeros(i + 1)

        for j in range(i + 1):
            log_probs[j] = log_cluster_prior_pred_prob(i + 1, j + 1, w, w_params)

        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        new_val = rng.choice(np.arange(i + 1),p = probs/np.sum(probs))

        if new_val != clusters_graph[i]:
            old_cluster_val = cluster_values[value_to_cluster[i]]
            old_tau = tau_sq[value_to_cluster[i]]

            if new_val == i:
                new_tau = invgamma.rvs(tau_a, scale = tau_b)
                #cov_matrix = np.identity(p)*0.5
                new_cluster_val = np.random.multivariate_normal(np.zeros(p), cov_matrix)
                
            else:
                new_cluster_val = cluster_values[value_to_cluster[new_val]]
                new_tau = tau_sq[value_to_cluster[new_val]]

            log_acc = 0.5*len(subtree)*(np.log(old_tau) - np.log(new_tau)) +np.sum(-1/(2*new_tau)*(y[subtree] - X[subtree]@new_cluster_val)**2 +1/(2*old_tau)*(y[subtree] - X[subtree]@old_cluster_val)**2)

            log_U = np.log(np.random.uniform())

            if log_U <= log_acc:
                if ret_graph[i] == i:
                    cluster_values.pop(value_to_cluster[i])
                    tau_sq.pop(value_to_cluster[i])

                ret_graph[i] = new_val

                if new_val == i:
                    subtree, value_to_cluster = construct_subtree(ret_graph, -1, n)
                    cluster_values.insert(value_to_cluster[i], new_cluster_val)
                    tau_sq.insert(value_to_cluster[i], new_tau)
        elif new_val == i:
            old_cluster_val = cluster_values[value_to_cluster[i]]
            old_tau = tau_sq[value_to_cluster[i]]
            new_tau = invgamma.rvs(tau_a, scale = tau_b)
            #cov_matrix = np.identity(p)*0.5
            new_cluster_val = np.random.multivariate_normal(np.zeros(p), cov_matrix)


            log_acc = 0.5*len(subtree)*(np.log(old_tau) - np.log(new_tau)) + np.sum(-1/(2*new_tau)*(y[subtree] - X[subtree]@new_cluster_val)**2 +1/(2*old_tau)*(y[subtree] - X[subtree]@old_cluster_val)**2)
            log_U = np.log(np.random.uniform())

            if log_U <= log_acc:
                cluster_values[value_to_cluster[i]] = new_cluster_val
                tau_sq[value_to_cluster[i]] = new_tau

    return cluster_values, tau_sq, ret_graph

#posterior tau sampling
def sample_posterior_tau(clusters : List[List[int]], y:np.ndarray, X:np.ndarray, beta : np.ndarray, init_a: float, init_b:float) -> float:
    n, p = X.shape
    num_clusters = len(clusters)
    cluster_tau = np.zeros(num_clusters)
    index = 0

    for cluster in clusters:
        b_arr = y[cluster] - X[cluster]@beta[index]
        cluster_tau[index] = (len(cluster) - 1)*invgamma.rvs(init_a + len(cluster)/2, scale= init_b + 1/2*np.inner(b_arr, b_arr))
        index += 1

    if n == num_clusters:
        return 1

    return 1/(n - num_clusters)*np.sum(cluster_tau)

#posterior sigma_beta sampling
def sample_posterior_sigma_beta(clusters : List[List[int]], beta:np.ndarray, init_a, init_b) -> float:
    num_clusters = len(clusters)
    b = 0

    for i in range(num_clusters):
        b += np.inner(beta[i], beta[i])

    return invgamma.rvs(init_a + num_clusters/2, scale = init_b + b/2)

# Sample predictive distribution
def sample_y_predictive(clusters: List[List[int]], w, prior_type, cluster_values: List, x:np.ndarray, n:int, Sigma:np.ndarray, tau_list:List, tau_a : float, tau_b:float):
    if prior_type == "beta":
        gos_params = Beta_Gos_params(w)
    elif prior_type == "gamma":
        gos_params = Gamma_GOS_params(w)
    else:
        gos_params = Dir_proc_params(1)
    
    p = Sigma.shape[0]
    num_clusters = len(clusters)
    probs = np.zeros(num_clusters + 1)
    probs[num_clusters] = np.exp(gos_params.log_r(n))

    for i in range(num_clusters):
        for j in clusters[i]:
            # for x_{n} we need p_{n - 1, i}
            probs[i] += np.exp(gos_params.log_p(n, j + 1))

    rng = np.random.default_rng()
    new_val = rng.choice(np.arange(num_clusters + 1),p = probs/np.sum(probs))

    if new_val == num_clusters:
        new_phi = np.random.multivariate_normal(np.zeros(p), Sigma)
        tau = invgamma.rvs(tau_a, scale = tau_b)
    else:
        new_phi = cluster_values[new_val]
        tau = tau_list[new_val]


    return np.random.normal(np.inner(x, new_phi), tau)
 
def test_predictive(iter, prior, alpha, alpha_const, X, y):
    n, p = X.shape

    predict = np.zeros(iter)

    for i in range(iter):
        result = gibbs_sampling(y, X, 5000, 2, np.var(y), np.zeros(p), 0.01, 0.01, 1, alpha, alpha_const, prior, True)
        cluster = construct_cluster_list(result["clusters"][-1], n)

        #Change to AR1 matrix 
        x = np.array([1, y[-1], y[-2]])
        predict[i] = sample_y_predictive(cluster, result["W"][-1], "gamma", result["cluster_values"][-1], x, n, np.identity(p), result["tau"][-1], 2, np.var(y))
    
    print(f"{alpha_const} {prior} mean : {np.mean(predict)}")

    return predict

if __name__ == "__main__":
    iterations = 100000

    data = np.load("covid_dat.npz")
    X = data["X"]
    y = data["y"]

    n, p = X.shape

    tests = [ [False, "beta", "long", 1], [True, "beta", "short", 2], [False, "gamma", "long", 3], [True, "gamma", "short", 5], [True, "dir", "short", 1]]
    

    for test in tests:
        result = gibbs_sampling(y, X, iterations, 2, np.var(y), np.zeros(p), 0.01, 0.01, 1, test[3], test[0], test[1], False)

        for key, value in result.items():
            print(f"{key} : {value[-1]}")

        last_list = construct_cluster_list(result["clusters"][-1], n)
        print(last_list)

        np.savez(f"covid_ar_non_conj/{test[2]}_mem_{test[1]}_single_params.npz", w = result["W"], X = X, y= y, rho = result["rho"])

        with open(f"covid_ar_non_conj/{test[2]}_mem_{test[1]}_clusters.pkl", "wb") as f:
            pickle.dump(result["clusters"], f)

        with open(f"covid_ar_non_conj/{test[2]}_mem_{test[1]}_cluster_values.pkl", "wb") as f:
            pickle.dump(result["cluster_values"], f, pickle.HIGHEST_PROTOCOL)

        with open(f"covid_ar_non_conj/{test[2]}_mem_{test[1]}_tau_values.pkl", "wb") as f:
            pickle.dump(result["tau"], f, pickle.HIGHEST_PROTOCOL)