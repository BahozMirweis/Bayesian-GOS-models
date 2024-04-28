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

    # Init tau
    tau = 1

    # Init sigma beta
    sigma_beta = 5

    Lambda = np.diag(np.full(p, tau/sigma_beta))

    params = {"cluster_values" : [], "W" : np.zeros((iter, n)), "tau" : np.zeros(iter), "clusters" : []}

    for i in tqdm(range(iter)):
        cluster_graph = sample_cluster_posterior(y, X, cluster_graph, init_cluster_beta_mu, Lambda, sigma_beta, tau, w, w_params)

        # Update w
        #w = sample_posterior_w_beta(cluster_graph, n, alpha, beta)

        if gos_prior == "gamma":
            w = sample_posterior_w_gamma(cluster_graph, w, n, alpha, beta, w_params)
        elif gos_prior == "beta":
            w = sample_posterior_w_beta(cluster_graph, n, alpha, beta)

        w_params.update_r(w)

        # update cluster values
        clusters = construct_cluster_list(cluster_graph, n)
        beta_cluster_value = sample_cluster_values(clusters, y, X, init_cluster_beta_mu, Lambda)

        # update tau
        tau = sample_posterior_tau(clusters, y, X, beta_cluster_value, init_tau_a, init_tau_b)

        # update sigma beta
        #sigma_beta = sample_posterior_sigma_beta(clusters, beta_cluster_value, init_sigma_beta_a, init_sigma_beta_b)

        Lambda = np.diag(np.full(p, tau/sigma_beta))

        #Add param updates
        params["cluster_values"].append(beta_cluster_value)
        params["W"][i] = w
        params["tau"][i] = tau
        params["clusters"].append(cluster_graph)
        cluster_graph = cluster_graph.copy()

    return params

@nb.njit(fastmath=True)
def matrix_invert2x2(mat : np.ndarray) -> np.ndarray:
    deter = mat[0,0]*mat[1, 1] - mat[0, 1]*mat[1, 0]
    invert = np.full((2,2), 1/deter)
    invert[0,0] *= mat[1, 1]
    invert[1, 1]*= mat[0, 0]
    invert[0, 1] *= -mat[0, 1]
    invert[1, 0] *= -mat[1, 0]

    return invert

@nb.njit(fastmath=True)
def matrix_det2x2(mat:np.ndarray) -> np.ndarray:
    return mat[0,0]*mat[1, 1] - mat[0, 1]*mat[1, 0]

@nb.njit(fastmath=True)
def matrix_inver3x3(m):    
    m1, m2, m3, m4, m5, m6, m7, m8, m9 = m.ravel()
    inv = np.array([[m5*m9-m6*m8, m3*m8-m2*m9, m2*m6-m3*m5],
                    [m6*m7-m4*m9, m1*m9-m3*m7, m3*m4-m1*m6],
                    [m4*m8-m5*m7, m2*m7-m1*m8, m1*m5-m2*m4]])
    return inv / np.dot(inv[0], m[:, 0])

@nb.njit(fastmath = True)
def matrix_det3x3(m):
    return m[0,0]*(m[1,1]*m[2, 2] - m[1, 2]*m[2, 1]) - m[0, 1]*(m[1, 0]*m[2, 2] - m[1, 2]*m[2, 0]) + m[0, 2]*(m[1, 0]*m[2,1] - m[2, 0]*m[1, 1])

def mat_inv(m):
    if m.shape[0] == 1:
        return 1/m
    if m.shape[0] == 2:
        return matrix_invert2x2(m)
    elif m.shape[0] == 3:
        return matrix_inver3x3(m)
    else:
        return np.linalg.inv(m)

def mat_det(m):
    if m.shape[0] == 1:
        return m[0, 0]
    if m.shape[0] == 2:
        return matrix_det2x2(m)
    elif m.shape[0] == 3:
        return matrix_det3x3(m)
    else:
        return np.linalg.det(m)

def log_marginal_likelihood(y:np.ndarray, X:np.ndarray,  clusters: List[List[int]], subtree : List[int], beta_mean:np.ndarray, Lambda : np.ndarray, sigma_beta_sq:float, tau_sq:float) -> np.ndarray:
    num_clusters = len(clusters)
    n, p = X.shape

    log_prob_init = -num_clusters/(2*sigma_beta_sq)*np.inner(beta_mean, beta_mean)

    log_probs_without_cluster = np.zeros(num_clusters)
    log_probs_with_cluster = np.zeros(num_clusters + 1)

    identity = np.identity(p)
    i = 0

    y_subtree = y[subtree]
    X_subtree = X[subtree]

    for cluster in clusters:
        cluster_X = X[cluster]
        cluster_y = y[cluster]

        cluster_Lambda = cluster_X.T @ cluster_X + Lambda
        # cluster_Lambda*mu
        cluster_mean_unscaled = Lambda@beta_mean.T +cluster_y@cluster_X
            
        log_probs_without_cluster[i] -= 1/2*np.log(mat_det(cluster_Lambda))
        log_probs_without_cluster[i] += 1/(2*tau_sq)*np.dot(cluster_mean_unscaled@mat_inv(cluster_Lambda), cluster_mean_unscaled.T)

        cluster_X_with_pos = X[cluster + subtree]
        cluster_mean_unscaled += y_subtree@X_subtree

        cluster_Lambda = cluster_X_with_pos.T @ cluster_X_with_pos + Lambda

        log_probs_with_cluster[i] -= 1/2*np.log(mat_det(cluster_Lambda))
        log_probs_with_cluster[i] += 1/(2*tau_sq)*np.dot(cluster_mean_unscaled@mat_inv(cluster_Lambda), cluster_mean_unscaled.T)
        
        i += 1

    # final cluster
    cluster_Lambda = X_subtree.T @ X_subtree + Lambda
    cluster_mean_unscaled = Lambda@beta_mean.T +y_subtree@X_subtree

    log_probs_with_cluster[num_clusters] -= 1/2*np.log(mat_det(cluster_Lambda))
    log_probs_with_cluster[num_clusters] += 1/(2*tau_sq)*np.dot(cluster_mean_unscaled@mat_inv(cluster_Lambda), cluster_mean_unscaled.T)
    # calc probs

    log_probs = np.zeros(num_clusters + 1)
    log_prob_without_pos = log_prob_init + np.sum(log_probs_without_cluster)
    log_probs[num_clusters] = log_prob_without_pos -(1 + num_clusters)/(2*sigma_beta_sq)*np.inner(beta_mean, beta_mean) + log_probs_with_cluster[num_clusters]

    for i in range(num_clusters):
        log_probs[i] = log_prob_without_pos - log_probs_without_cluster[i] + log_probs_with_cluster[i]

    return log_probs


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

def sample_cluster_posterior(y:np.ndarray, X:np.ndarray,  clusters_graph: dict[int, int], beta_mean:np.ndarray, Lambda: np.ndarray, sigma_beta_sq:float, tau_sq:float, w:np.ndarray, w_params) -> dict[int, int]:
    n, p = X.shape
    ret_graph = clusters_graph.copy()
    rng = np.random.default_rng()

    for i in range(n):
        clusters, subtree, value_to_cluster = construct_cluster_list_without_pos(ret_graph, i, n)
        log_lik_probs = log_marginal_likelihood(y, X, clusters, subtree, beta_mean, Lambda, sigma_beta_sq, tau_sq)
        log_probs = np.zeros(i + 1)
        value_to_cluster[i] = len(clusters)

        #cluster_probs = np.zeros(i + 1)

        for j in range(i + 1):
            #cluster_probs[j] = log_cluster_prior_pred_prob(i + 1, j + 1, w)
            log_probs[j] = log_lik_probs[value_to_cluster[j]] + log_cluster_prior_pred_prob(i + 1, j + 1, w, w_params)

        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        new_val = rng.choice(np.arange(i + 1),p = probs/np.sum(probs))
        ret_graph[i] = new_val

    return ret_graph

# Posterior cluster value sampling
def sample_cluster_values(clusters : List[List[int]], y:np.ndarray, X:np.ndarray, beta_mean:np.ndarray, Lambda: np.ndarray) -> np.ndarray:
    n, p = X.shape
    num_clusters = len(clusters)
    beta = np.zeros((num_clusters, p))
    identity = np.identity(p)
    index = 0

    for cluster in clusters:
        cluster_X = X[cluster]
        cluster_y = y[cluster]

        cluster_Lambda = cluster_X.T @ cluster_X + Lambda
        inv_cluster_lambda = mat_inv(cluster_Lambda)

        cluster_mean = inv_cluster_lambda@(Lambda@beta_mean.T +cluster_y@cluster_X)

        beta[index] = np.random.multivariate_normal(cluster_mean, inv_cluster_lambda)
        index += 1

    return beta

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

    return 1/(n - num_clusters)*np.sum(cluster_tau)

#posterior sigma_beta sampling
def sample_posterior_sigma_beta(clusters : List[List[int]], beta:np.ndarray, init_a, init_b) -> float:
    num_clusters = len(clusters)
    b = 0

    for i in range(num_clusters):
        b += np.inner(beta[i], beta[i])

    return invgamma.rvs(init_a + num_clusters/2, scale = init_b + b/2)



if __name__ == "__main__":

    iterations = 10000

    data = np.load("lin_data.npz")
    X = data["X"]
    y = data["y"]
    n, p = X.shape

    result = gibbs_sampling(y, X, iterations, 0.01, 0.01, np.zeros(2), 0.01, 0.01, 1, 1, False, "beta")

    for key, value in result.items():
        print(f"{key} : {value[-1]}")

    last_list = construct_cluster_list(result["clusters"][-1], n)
    print(last_list)

    np.savez("linear_reg_results/long_mem_beta_single_params.npz", w = result["W"], tau = result["tau"], X = X, y= y)

    with open("linear_reg_results/long_mem_beta_clusters.pkl", "wb") as f:
        pickle.dump(result["clusters"], f)

    with open("linear_reg_results/long_mem_beta_cluster_values.pkl", "wb") as f:
        pickle.dump(result["cluster_values"], f, pickle.HIGHEST_PROTOCOL)
    
