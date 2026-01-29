# Library of functions for defining a multidimensional environment with 
# discrete features. Probability of reward is highest for a target feature 
# that changes with probability h.

import numpy as np

class World(object):
	"""Container for world that the agent is learning in.

	Parameters
	----------
	n_dims : int
	    Number of dimensions along which world varies.
	n_feats_per_dim : int
	    Number of discrete features per dimension.
	n_feats : int
	    Number of total features
	obs_space: array, n_dims by n_feats per dim, int
	    Array corresponding to full observation space.
	target: int
		Target feature.
	outcome: float
		Outcome value.
	h: float
		Rate of change for target.
	p_high: float
		Rate of reward given target.
	p_low: float
		Rate of reward not given target.
	"""

	def __init__(self, n_dims, n_feats_per_dim, h, p_high, p_low, outcome, target=None):

        ## Define observation space.
		self.n_dims = n_dims
		self.n_feats_per_dim = n_feats_per_dim
		self.n_feats = n_dims * n_feats_per_dim
		# Make array corresponding to full observation space.
		os = np.zeros([n_dims, n_feats_per_dim])
		for d in np.arange(0, n_dims)+1:
			if d == 1:
				os[d-1,] = np.arange(0, n_feats_per_dim)+1
			else:
				os[d-1,] = np.arange(1, n_feats_per_dim+1) + (d-1)*n_feats_per_dim
		self.obs_space = os

        ## Define state model.
        # If not provided, randomly select relevant feature 
		if target is None:
			self.target = np.random.randint(self.n_feats)+1
		else:
			self.target = target
        # Probability with which target feature changes
		self.h = h

		# Keep track of target history (useful when h > 0)
		self.target_history = [self.target]

        ## Define reward function.
        # Probability of observing outcome given target feature is present
		self.p_high = p_high
        # Probability of observing outcome given target feature is not present
		self.p_low = p_low
        # Magnitude of binary outcome
		self.outcome = outcome

	def maybe_update_target(self):

		"""With probability h, switch the target feature to a different feature.

		This implements a non-stationary reward function where the relevant
		target feature can change over time. Should typically be called
		once per trial before generating an outcome.
		"""

		# No changes if h <= 0 (stationary target)
		if self.h <= 0:
			return

		# With probability h, change the target to a different feature
		if np.random.rand() < self.h:
			all_feats = np.arange(1, self.n_feats + 1)
			candidates = all_feats[all_feats != self.target]
			self.target = int(np.random.choice(candidates))

		# Record current target (even if unchanged) for analysis/plotting
		self.target_history.append(self.target)

	def plot_properties(self):

		"""Plot reward and stay probability distributions in feature space.
		These will depend on the properties of the world (i.e. the reward
		function and change model). 

	    Returns
	    -------
	    fig, ax : plt.figure
	        Figure and axis of plot.
	      
	    Notes
	    -----
	    Requires matplotlib.
	    """

		import matplotlib.pyplot as plt
		import matplotlib.cm as cm
		from mpl_toolkits.axes_grid1 import make_axes_locatable

		ratio = float(self.n_dims) / float(self.n_feats_per_dim)

		## Prepare probability distributions for visualization
		# Reward probability
		R = (self.obs_space == self.target).astype(float) 
		R[R == 1] = self.p_high
		R[R == 0] = self.p_low

		# Stay probability
		S = (self.obs_space == self.target).astype(float) 
		S[S == 1] = 1 - self.h
		S[S == 0] = self.h / (self.n_feats-1)

		## Plot probability distributions.
		# Prepare all axes 
		fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize=(8, 4));
		div1 = make_axes_locatable(ax[0])
		div2 = make_axes_locatable(ax[1])
		cax1 = div1.append_axes("right", size="5%", pad=0.05)
		cax2 = div2.append_axes("right", size="5%", pad=0.05)
		# Probability of reward in feature space.
		im1 = ax[0].imshow(R.T, cmap=cm.gray, aspect=ratio, clim = (0, 1))
		ax[0].set_title('Reward probability',fontsize = 20);
		ax[0].set_xticks([]);
		ax[0].set_yticks([]);
		ax[0].set_xlabel('Dimension',fontsize = 15)
		ax[0].set_ylabel('Feature',fontsize = 15)
		plt.colorbar(im1, cax=cax1)
		# Probability of switching in feature space.
		im2 = ax[1].imshow(S.T, cmap=cm.gray, aspect=ratio, clim = (0, 1))
		ax[1].set_title('Switch probability',fontsize = 20);
		ax[1].set_xticks([]);
		ax[1].set_yticks([]);
		ax[1].set_xlabel('Dimension',fontsize = 15)
		ax[1].set_ylabel('Feature',fontsize = 15)
		plt.colorbar(im2, cax=cax2)

		return fig, ax

	def plot_reward_function(self):
		"""Plot detailed visualization of the reward function structure.
		
		Creates three panels showing:
		1. Heatmap of reward probabilities by dimension and feature
		2. Bar chart of reward probabilities for each feature number
		3. Comparison of expected reward for target-present vs target-absent stimuli
		
		Returns
		-------
		fig, axes : plt.figure, array of axes
			Figure and axes of the plot.
		
		Notes
		-----
		Requires matplotlib.
		"""
		
		import matplotlib.pyplot as plt
		from matplotlib.patches import Rectangle
		
		# Create figure with 3 subplots
		fig, axes = plt.subplots(1, 3, figsize=(16, 4))
		
		# 1. Heatmap of reward probabilities by feature
		# Create reward probability matrix
		reward_matrix = np.zeros((self.n_dims, self.n_feats_per_dim))
		for d in range(self.n_dims):
			for f in range(self.n_feats_per_dim):
				feat_num = self.obs_space[d, f]  # Feature number in expanded coding
				if feat_num == self.target:
					reward_matrix[d, f] = self.p_high
				else:
					reward_matrix[d, f] = self.p_low
		
		# Plot heatmap using muted blue/red from vlag colormap (consistent with choice visualization)
		vlag_colors = plt.cm.RdYlBu_r  # Reversed to get blue-red
		im1 = axes[0].imshow(reward_matrix, cmap=vlag_colors, aspect='auto', vmin=0, vmax=1)
		axes[0].set_title('Reward Probability by Feature\n(Dimension × Feature)', fontsize=12, fontweight='bold')
		axes[0].set_xlabel('Feature Index', fontsize=11)
		axes[0].set_ylabel('Dimension', fontsize=11)
		axes[0].set_xticks(range(self.n_feats_per_dim))
		axes[0].set_xticklabels([f'F{i+1}' for i in range(self.n_feats_per_dim)])
		axes[0].set_yticks(range(self.n_dims))
		axes[0].set_yticklabels([f'Dim {i+1}' for i in range(self.n_dims)])
		plt.colorbar(im1, ax=axes[0], label='Reward Probability')
		
		# Add text annotations
		for d in range(self.n_dims):
			for f in range(self.n_feats_per_dim):
				feat_num = self.obs_space[d, f]
				text_color = 'white' if reward_matrix[d, f] > 0.5 else 'black'
				axes[0].text(f, d, f'Feat {feat_num}', 
				             ha='center', va='center', color=text_color, fontweight='bold')
				if feat_num == self.target:
					# Highlight target feature
					rect = Rectangle((f-0.5, d-0.5), 1, 1, fill=False, 
					               edgecolor='black', linewidth=3)
					axes[0].add_patch(rect)
		
		# 2. Bar chart of reward probabilities for each feature
		# Use muted red/blue from vlag colormap (consistent with choice visualization)
		vlag_colors = plt.cm.RdYlBu_r  # Reversed to get blue-red
		reward_color = vlag_colors(0.2)  # Muted blue
		no_reward_color = vlag_colors(0.8)  # Muted red
		
		feature_nums = np.arange(1, self.n_feats + 1)
		reward_probs = np.array([self.p_high if f == self.target else self.p_low 
		                         for f in feature_nums])
		colors = [reward_color if f == self.target else no_reward_color for f in feature_nums]
		
		bars = axes[1].bar(feature_nums, reward_probs, color=colors, alpha=0.7, edgecolor='black')
		axes[1].axhline(y=self.p_high, color=reward_color, linestyle='--', alpha=0.5, label=f'High ({self.p_high})')
		axes[1].axhline(y=self.p_low, color=no_reward_color, linestyle='--', alpha=0.5, label=f'Low ({self.p_low})')
		axes[1].set_title('Reward Probability by Feature Number', fontsize=12, fontweight='bold')
		axes[1].set_xlabel('Feature Number', fontsize=11)
		axes[1].set_ylabel('Reward Probability', fontsize=11)
		axes[1].set_ylim([0, 1])
		axes[1].set_xticks(feature_nums)
		axes[1].legend()
		axes[1].grid(True, alpha=0.3, axis='y')
		
		# Highlight target feature
		target_bar = bars[self.target - 1]
		target_bar.set_edgecolor('black')
		target_bar.set_linewidth(3)
		axes[1].text(self.target, self.p_high + 0.05, 'TARGET', 
		            ha='center', fontweight='bold', fontsize=10)
		
		# 3. Expected reward for different stimulus types
		# Show how reward probability depends on whether target feature is present
		# Use muted red/blue from vlag colormap (consistent with choice visualization)
		vlag_colors = plt.cm.RdYlBu_r  # Reversed to get blue-red
		reward_color = vlag_colors(0.2)  # Muted blue
		no_reward_color = vlag_colors(0.8)  # Muted red
		
		stim_types = ['Target\nPresent', 'Target\nAbsent']
		expected_rewards = [self.p_high, self.p_low]
		colors_stim = [reward_color, no_reward_color]
		
		bars2 = axes[2].bar(stim_types, expected_rewards, color=colors_stim, alpha=0.7, 
		                   edgecolor='black', linewidth=2)
		axes[2].set_title('Expected Reward by Stimulus Type', fontsize=12, fontweight='bold')
		axes[2].set_ylabel('Reward Probability', fontsize=11)
		axes[2].set_ylim([0, 1])
		axes[2].grid(True, alpha=0.3, axis='y')
		
		# Add value labels on bars
		for i, (bar, val) in enumerate(zip(bars2, expected_rewards)):
			axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.02, 
			            f'{val:.2f}', ha='center', fontweight='bold', fontsize=11)
		
		plt.tight_layout()
		
		return fig, axes

	def make_stimuli(self):

		"""Returns one instance of all possible stimuli given the feature space, coded in 
		two ways.

		Returns
	    -------
	    stimuli1 : array, int, shape(n_dims, n_feats_per_dim)
	        Rows are dimensions and columns are features. Non-expanded coding.

	    stimuli2 : array, int, shape(n_dims, n_feats_per_dim)
	    	Same as above, expanded coding (features are labeled 1-n_feats).

	    single_stim : array, int, shape(1, n_feats_per_dim)
	   		Randomly selected single stimulus.

		"""

		stimuli1 = np.random.permutation(np.arange(self.n_feats_per_dim)+1)
		stimuli2 = stimuli1

		## Loop through dimensions.
		for d in np.arange(self.n_dims-1)+1:

			## Permute features.
			new = np.random.permutation(np.arange(self.n_feats_per_dim)+1)

			## Stack on top previous features.
			stimuli1 = np.vstack((stimuli1,new))
			stimuli2 = np.vstack((stimuli2, new + self.n_feats_per_dim*(d)))

		## Draw one stimulus at random.
		a_rand = np.random.randint(1,self.n_feats_per_dim+1)
		single_stim = stimuli2[:,a_rand-1]	

		return stimuli1, stimuli2, single_stim

	def make_observations(self, n_trials):
		""" Convenience function for generating random observation sequences of a 
			given length. Each observation consists of a [stimulus, outcome] pair. 
		"""

		stimuli = np.empty((0,self.n_feats_per_dim))
		outcomes = np.empty(n_trials)

		## Loop through trials. 
		for t in np.arange(n_trials):
			
			## Generate a random stimulus. 
			d, e, stimulus = self.make_stimuli()
			stimuli = np.vstack((stimuli, stimulus))

			## Compute outcome. 
			outcomes[t] = self.generate_outcome(stimulus)

		return stimuli, outcomes

	def plot_observations(self, stimuli, outcomes, target=None):
		""" Plots sequence of observations (choices and outcomes).
			Can be used with both simulated and real data.
		"""

		import matplotlib.pyplot as plt
		import seaborn as sns

		n_trials, d = stimuli.shape

		## If target is not provided, get default from world.
		if target is None: target = self.target

		fig, ax = plt.subplots(1, 1, figsize=(30,6));
		win_col = '#4dac26'
		loss_col = '#bababa'
		sz_targ = 16
		sz_nontarg = 16
		# plt.axhline(y=target, color='#f0f0f0', linestyle='-',linewidth='18')
		for t in np.arange(n_trials):
		    if outcomes[t] == 1:
		        for d in np.arange(self.n_dims):
		            if stimuli[t,d] == target:
		                plt.plot(t+1, stimuli[t,d],'s', color=win_col, markersize=sz_targ, markeredgecolor='#252525', markeredgewidth=2)
		            else:
		                plt.plot(t+1, stimuli[t,d],'s', color=win_col, markersize=sz_nontarg)    
		    else:
		        for d in np.arange(self.n_dims):
		            if stimuli[t,d] == target:
		                plt.plot(t+1, stimuli[t,d],'s', color=loss_col, markersize=sz_targ, markeredgecolor='#252525', markeredgewidth=2)      
		            else:
		                plt.plot(t+1, stimuli[t,d],'s', color=loss_col, markersize=sz_nontarg) 

		ax.set_xlim((0, n_trials+1));
		ax.set_ylim((0, self.n_feats+2));
		ax.set_xticks(np.arange(n_trials)+1);
		ax.set_yticks(np.arange(self.n_feats)+1);
		xl = ax.set_xlabel('Trial',fontsize = 30);
		yl = ax.set_ylabel('Feature',fontsize = 30);                
		ax.tick_params(labelsize=30)
		plt.ylim([0,self.n_feats+1])
		# sns.despine()

		return fig, ax

	def generate_outcome(self, stimulus, target=None):

		"""Generates binary outcome given stimulus. Requires "stimuli2" coding from
		make_stimuli
		
		Parameters
	    -------
	    stimulus : array, int, shape(n_dims, 1)
	    	Expanded coding.
	    
	    target: int
	    	If target feature not provided, uses default from self.
	    
		Returns
	    -------
	    outcome : int, 0 or 1
	        
		"""

		## If target is not provided, possibly update it according to hazard rate h
		# (non-stationary world), then use the world's current target.
		if target is None:
			self.maybe_update_target()
			target = self.target
		
		## Compute outcome.
		if target in stimulus:
			outcome = int((np.random.rand() < self.p_high))
		else:
			outcome = int((np.random.rand() < self.p_low))

		return outcome





		

