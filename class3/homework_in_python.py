import numpy as np

def f_fitRL_rw(params, data, start_values=None):
    """
    Run a simple Rescorla-Wagner RL model on trial data.

    Parameters
    ----------
    params : array-like
        [lr, itemp]
        lr    : learning rate
        itemp : inverse temperature for softmax
    data : array-like, shape (n_trials, 3)
        Columns: [trial, chosen_stimulus, reward]
        - chosen_stimulus is assumed to be 1, 2, or 3 (like in the MATLAB code).
    start_values : array-like, optional
        Initial values for each stimulus, length nStim.

    Returns
    -------
    ll : float
        Negative log-likelihood of the observed choices under the model.
    pChoice : ndarray, shape (n_trials, nStim)
        Softmax choice probabilities over stimuli at each trial.
    value : ndarray, shape (n_trials + 1, nStim)
        Value estimates for each stimulus (includes initial + all updated values).
    choiceValue : ndarray, shape (n_trials, nStim)
        Values of options ordered: [chosen, unchosen1, unchosen2].
    choiceProbs : ndarray, shape (n_trials, nStim)
        Choice probs ordered: [chosen, unchosen1, unchosen2].
    """
    data = np.asarray(data)
    lr = float(params[0])
    itemp = float(params[1])

    n_trials = data.shape[0]
    # Stimulus IDs assumed to be integers like 1, 2, 3...
    stim_ids = np.unique(data[:, 1]).astype(int)
    nStim = len(stim_ids)

    # Preallocate arrays
    # value needs an extra row for the last updated values (trial+1)
    value = np.zeros((n_trials + 1, nStim), dtype=float)
    pChoice = np.zeros((n_trials, nStim), dtype=float)
    choiceValue = np.zeros((n_trials, nStim), dtype=float)
    choiceProbs = np.zeros((n_trials, nStim), dtype=float)

    # If start values are passed, set the first row
    if start_values is not None:
        start_values = np.asarray(start_values, dtype=float)
        if start_values.shape[0] != nStim:
            raise ValueError("start_values must have length equal to number of stimuli.")
        value[0, :] = start_values

    # Initialize negative log-likelihood
    ll = 0.0

    # Assume exactly 3 stimuli as in MATLAB (stimOptions = [1,2,3])
    # but we’ll compute on indices (0,1,2) internally
    stimOptions_idx = np.arange(nStim)  # 0,1,...,(nStim-1)

    for t in range(n_trials):
        # Current values for softmax
        v_t = value[t, :]

        # Softmax (numerically stable)
        z = itemp * v_t
        z -= np.max(z)          # for numerical stability
        exp_z = np.exp(z)
        p = exp_z / np.sum(exp_z)

        pChoice[t, :] = p

        # Chosen stimulus ID (1-based in data), convert to 0-based index
        chosenStim_id = int(data[t, 1])
        chosen_idx = chosenStim_id - 1

        # Unchosen indices
        unchosen_idx = stimOptions_idx[stimOptions_idx != chosen_idx]
        unChosenStim1_idx = unchosen_idx[0]
        unChosenStim2_idx = unchosen_idx[1]

        # Save values ordered by [chosen, unchosen1, unchosen2]
        choiceValue[t, 0] = v_t[chosen_idx]
        choiceValue[t, 1] = v_t[unChosenStim1_idx]
        choiceValue[t, 2] = v_t[unChosenStim2_idx]

        # Save probs ordered by [chosen, unchosen1, unchosen2]
        choiceProbs[t, 0] = p[chosen_idx]
        choiceProbs[t, 1] = p[unChosenStim1_idx]
        choiceProbs[t, 2] = p[unChosenStim2_idx]

        # Accumulate negative log-likelihood
        ll -= np.log(p[chosen_idx] + 1e-15)  # small epsilon to avoid log(0)

        # Update values using delta rule
        reward = data[t, 2]
        value[t + 1, :] = value[t, :]  # carry over previous values
        value[t + 1, chosen_idx] = value[t, chosen_idx] + lr * (reward - value[t, chosen_idx])

    return ll, pChoice, value, choiceValue, choiceProbs


def fitRL_rw(params, data):
    ll, pChoice, value, choiceValue, choiceProbs = f_fitRL_rw(params, data)
    return ll

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize



# ------------------------------------------------------------------
# Load in the data
# ------------------------------------------------------------------
class3data = np.loadtxt("class3data_homework.txt")
import 
# Datafile information:
# Columns:
# 0 - Subject (1-n)
# 1 - Schedule (ignore for now)
# 2 - Session Number (1-8)
# 3 - Trial Number (1-300)
# 4 - Choice (0-2: three buttons)
# 5 - Reward (0-1)
# 6 - ReactionTime (0-30000 ms)

# The choice numbers in the data file are 0 - 2 (needs to be 1-3 to map to columns)
class3data[:, 4] = class3data[:, 4] + 1

# You want to run the analysis on trials, choices and rewards
# For simplicity select these three columns (trial, choice, reward)
schedule_data = class3data[:, 3:6]

# ------------------------------------------------------------------
# Optimization setup (analog to fminsearch in MATLAB)
# ------------------------------------------------------------------
# Initial parameter guess [LR, IT]
params0 = np.array([0.5, 5.0])

# Objective function for minimize
def objective(params):
    # fitRL_rw should return the negative log-likelihood (ll)
    return fitRL_rw(params, schedule_data)

# Max function evaluations / tolerances (similar to optimset)
options = {
    "maxfev": 1000,
    "xatol": 1e-5,
    "fatol": 1e-5,
}

# Unconstrained optimization (like fminsearch)
result = minimize(
    objective,
    params0,
    method="Nelder-Mead",
    options=options,
)

mparams = result.x
# lla = result.fun  # if you want the final ll from the optimizer

# Save the fitted parameter estimates for output
LR = mparams[0]
IT = mparams[1]

# ------------------------------------------------------------------
# Run the RL model using fitted parameters
# ------------------------------------------------------------------
ll, pChoice, pStimvalue, choiceValue, choiceProbs = f_fitRL_rw(mparams, schedule_data)

LL = ll  # log likelihood from best parameters

# ------------------------------------------------------------------
# Compute AIC and BIC
# ------------------------------------------------------------------
k = len(mparams)              # number of parameters
n = schedule_data.shape[0]    # number of trials

AIC = 2 * LL + 2 * k
BIC = np.log(n) * k + 2 * LL  # matches your MATLAB formula

print(f"Fitted LR = {LR:.4f}, IT = {IT:.4f}")
print(f"LL  = {LL:.4f}")
print(f"AIC = {AIC:.4f}")
print(f"BIC = {BIC:.4f}")

# ------------------------------------------------------------------
# Single session trial-by-trial plots
# ------------------------------------------------------------------
data_length = 300  # check your data length

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 1) Stimulus values
ax = axes[0, 0]
# pStimvalue is (n_trials+1, nStim); we’ll plot first 300 trials
ax.plot(pStimvalue[:data_length, :])
ax.set_title("Stimulus Values")
ax.set_xlim(0, data_length - 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Trials")
ax.set_ylabel("Modeled Value")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 2) Choice probabilities of each option
ax = axes[0, 1]
ax.plot(pChoice[:data_length, :])
ax.set_title("Choice Probabilities")
ax.set_xlim(0, data_length - 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Trials")
ax.set_ylabel("Choice Probability")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 3) Choice probability of chosen option (column 0 in choiceProbs)
ax = axes[1, 0]
ax.plot(choiceProbs[:data_length, 0])
ax.set_title("Choice Probability of Chosen Option")
ax.set_xlim(0, data_length - 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Trials")
ax.set_ylabel("Choice Probability")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 4) Choice of best model option (smoothed correctness)
ax = axes[1, 1]
correctChoice = (
    choiceProbs[:, 0] - np.max(choiceProbs[:, 1:3], axis=1) > 0
).astype(float)

# moving average smoothing window of size 5 (like smooth(...,'moving'))
window = 5
kernel = np.ones(window) / window
smoothed = np.convolve(correctChoice, kernel, mode="same")

ax.plot(smoothed[:data_length])
ax.set_title("Choice of Best Model Option")
ax.set_xlim(0, data_length - 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Trials")
ax.set_ylabel("Correct Choice (0 or 1)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()


