import numpy as np

def init_sigma(N):
    """
    Initialize the spin configuration for the Sherrington-Kirkpatrick model.

    Parameters
    ----------
    N : int
        The number of spins.

    Returns
    -------
    numpy.ndarray
        The spin configuration.
    """
    return np.random.choice([-1, 1], N)


def init_h(N, beta, random_state=None, ):
    """
    Initialize the external fields for the Sherrington-Kirkpatrick model.

    Parameters
    ----------
    N : int
        The number of spins.
    beta : float
    random_state : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        The external fields.
    """
    rng = np.random.default_rng(random_state)
    sig_h = np.sqrt(1 - beta)
    return rng.normal(0.0, sig_h, N)


def init_J(N, beta, rho, random_state=None,):
    """
    Initialize the coupling matrix for the Sherrington-Kirkpatrick model with sparsity.

    Parameters
    ----------
    N : int
        The number of spins.
    beta : float, optional
        Inverse temperature parameter.
    rho : float
        Fraction of non-zero elements in the coupling matrix (0 < rho ≤ 1).
    random_state : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        The symmetric coupling matrix Jij with sparsity controlled by rho.
    """
    if not (0 < rho <= 1):
        raise ValueError("rho must be between 0 (exclusive) and 1 (inclusive).")
    rng = np.random.default_rng(random_state)
    sig_J = np.sqrt(beta / (N * rho))  # Adjusted standard deviation for sparsity
    # Initialize an empty upper triangular matrix (excluding diagonal)
    J_upper = np.zeros((N, N))
    # Total number of upper triangular elements excluding diagonal
    total_elements = N * (N - 1) // 2
    # Number of non-zero elements based on rho
    num_nonzero = int(np.floor(rho * total_elements))
    if num_nonzero == 0 and rho > 0:
        num_nonzero = 1  # Ensure at least one non-zero element if rho > 0
    # Get the indices for the upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(N, k=1)
    # Randomly select indices to assign non-zero Gaussian values
    selected_indices = rng.choice(total_elements, size=num_nonzero, replace=False)
    # Map the selected flat indices to row and column indices
    rows = triu_indices[0][selected_indices]
    cols = triu_indices[1][selected_indices]
    # Assign Gaussian-distributed values to the selected positions
    J_upper[rows, cols] = rng.normal(loc=0.0, scale=sig_J, size=num_nonzero)
    # Symmetrize the matrix to make Jij symmetric
    Jij = J_upper + J_upper.T

    return Jij


def calc_basic_lfs(sigma, h, J):
    """
    Calculate the local fields for the Sherrington-Kirkpatrick model.

    Parameters
    ----------
    sigma : numpy.ndarray
        The spin configuration.
    h : numpy.ndarray
        The external fields.
    J : numpy.ndarray
        The coupling matrix.

    Returns
    -------
    numpy.ndarray
        The local fields.
        Divide by 2 because every term appears twice in symmetric case.
    """
    return h + 0.5 * J @ sigma


def calc_energies(sigma, h, J):
    """
    Calculate the energy delta of the system.

    Parameters
    ----------
    sigma : numpy.ndarray
        The spin configuration.
    h : numpy.ndarray
        The external fields.
    J : numpy.ndarray
        The coupling matrix.

    Returns
    -------
    numpy.ndarray
        The kis of the system.
    """
    return sigma * calc_basic_lfs(sigma, h, J)


def calc_dfe(sigma, h, J):
    """
    Calculate the distribution of fitness effects.

    Parameters
    ----------
    sigma : numpy.ndarray
        The spin configuration.
    h : numpy.ndarray
        The external fields.
    J : numpy.ndarray
        The coupling matrix.

    Returns
    -------
    numpy.ndarray
        The normalized distribution of fitness effects.
    """
    return -2 * calc_energies(sigma, h, J)


def calc_bdfe(sigma, h, J):
    """
    Calculate the Beneficial distribution of fitness effects.

    Parameters
    ----------
    sigma : numpy.ndarray
        The spin configuration.
    h : numpy.ndarray
        The external fields.
    J : numpy.ndarray
        The coupling matrix.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The beneficial fitness effects and the indices of the beneficial mutations.
    """
    DFE = calc_dfe(sigma, h, J)
    r = calc_rank(sigma, h, J)
    BDFE, b_ind = DFE[DFE >= 0], np.where(DFE > 0)[0]
    # Normalize the beneficial effects
    return BDFE, b_ind


def calc_rank(sigma, h, J):
    """
    Calculate the rank of the spin configuration.

    Parameters
    ----------
    sigma : numpy.ndarray
        The spin configuration.
    h : numpy.ndarray
        The external fields.
    J : numpy.ndarray
        The coupling matrix.

    Returns
    -------
    int
        The rank of the spin configuration.
    """
    dfe = calc_dfe(sigma, h, J)
    return np.sum(dfe > 0)


def sswm_flip(sigma, his, Jijs):
    """
    Choose a spin to flip using probabilities of the sswm regime probabilities.

    Parameters
    ----------
    sigma : numpy.ndarray
        The spin configuration.
    his : numpy.ndarray
        The local fitness fields.
    Jijs : numpy.ndarray

    Returns
    -------
    int : The index of the spin to flip.
    """
    effects, indices = calc_bdfe(sigma, his, Jijs)
    effects /= np.sum(effects)
    return np.random.choice(indices, p=effects)


def calc_F_off(sigma_init, his, Jijs):
    """
    Calculate the fitness offset for the given configuration.

    Parameters
    ----------
    sigma_init : numpy.ndarray
        The initial spin configuration.
    his : numpy.ndarray
        The local fitness fields.
    Jijs : numpy.ndarray
        The coupling matrix.

    Returns
    -------
    float
        The fitness offset.
    """
    return compute_fit_slow(sigma_init, his, Jijs) - 1


def compute_fit_slow(sigma, his, Jijs, F_off=0.0):
    """
    Compute the fitness of the genome configuration sigma using full slow computation.

    Parameters:
    sigma (np.ndarray): The genome configuration (vector of -1 or 1).
    his (np.ndarray): The vector of site-specific contributions to fitness.
    Jijs (np.ndarray): The interaction matrix between genome sites.
    F_off (float): The fitness offset, defaults to 0.

    Returns:
    float: The fitness value for the configuration sigma.
    Divide by 2 because every term appears twice in symmetric case.
    """
    return sigma @ (his + 0.5 * Jijs @ sigma) - F_off


def compute_fitness_delta_mutant(sigma, his, Jijs, k):
    """
    Compute the fitness change for a mutant at site k.

    Parameters:
    sigma (np.ndarray): The genome configuration.
    hi (np.ndarray): The vector of site-specific fitness contributions.
    f_i (np.ndarray): The local fitness fields.
    k (int): The index of the mutation site.

    Returns:
    float: The change in fitness caused by a mutation at site k.
    Divide by 2 because every term appears twice in symmetric case.
    """

    return -2 * sigma[k] * (his[k] + 0.5 * Jijs[k] @ sigma)


def relax_sk(sigma, his, Jijs):
    """
    Relax the Sherrington-Kirkpatrick model with given parameters.
    Parameters
    ----------
    sigma: numpy.ndarray
    his: numpy.ndarray
    Jijs: numpy.ndarray

    Returns
    -------
    list
        The mutation sequence.
    """
    flip_sequence = []
    rank = calc_rank(sigma, his, Jijs)

    while rank > 0:
        flip_idx = sswm_flip(sigma, his, Jijs)
        sigma[flip_idx] *= -1
        flip_sequence.append(flip_idx)
        rank = calc_rank(sigma, his, Jijs)

    return flip_sequence


def compute_sigma_from_hist(sigma_0, hist, num_muts=None):
    """
    Compute sigma from the initial sigma and the flip history up to num_muts mutations.

    Parameters
    ----------
    sigma_0 : numpy.ndarray
        The initial spin configuration.
    hist : list of int
        The flip history.
    num_muts : int
        The number of mutations to consider.

    Returns
    -------
    numpy.ndarray
        The spin configuration after num_muts mutations.
    """
    sigma = sigma_0.copy()
    if num_muts is None:
        rel_hist = hist
    else:
        rel_hist = hist[:num_muts]
    for flip in rel_hist:
        sigma[flip] *= -1
    return sigma

def curate_sigma_list(sigma_0, hist, flips):
    """
    Curate the sigma list to have num_points elements.

    Parameters
    ----------
    sigma_0 : numpy.ndarray
        The initial spin configuration.
    hist : list of int
        The flip history.
    flips : list of int
        The points to curate.

    Returns
    -------
    list
        The curated list of spin configurations.
    """
    sigma_list = []
    for flip in flips:
        sigma_t = compute_sigma_from_hist(sigma_0, hist, flip)
        sigma_list.append(sigma_t)
    return sigma_list
