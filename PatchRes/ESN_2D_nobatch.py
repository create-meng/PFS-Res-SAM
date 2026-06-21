"""Python version of D2_ESN (bidirectional Echo State Network for 2D data)."""

import torch
import numpy as np

device = torch.device("cpu")


class ESN_2D(torch.nn.Module):
    def __init__(self, input_dim=1, n_reservoir=100, spectral_radius=(0.9, 0.9), alpha=0.9,
                 connectivity=0.1, noise_level=1e-4, start_node=(1, 1), activation=torch.tanh, device_override=None):
        super(ESN_2D, self).__init__()
        self._device = device_override if device_override is not None else device
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.alpha = alpha
        self.connectivity = connectivity
        self.noise_level = noise_level
        self.start_node = start_node
        self.activation = activation

        W_in = 2 * np.random.rand(input_dim, n_reservoir) - 1
        W_res_1 = np.random.rand(n_reservoir, n_reservoir) - 0.5
        mask = np.random.rand(n_reservoir, n_reservoir) < connectivity
        W_res_1 *= mask
        W_res_2 = np.random.rand(n_reservoir, n_reservoir) - 0.5
        mask = np.random.rand(n_reservoir, n_reservoir) < connectivity
        W_res_2 *= mask

        rho_W_res = max(abs(np.linalg.eig(W_res_1)[0]))
        W_res_1 *= self.spectral_radius[0] / rho_W_res
        rho_W_res = max(abs(np.linalg.eig(W_res_2)[0]))
        W_res_2 *= self.spectral_radius[1] / rho_W_res

        self.W_in = torch.nn.Parameter(
            torch.from_numpy(W_in).float().to(self._device), requires_grad=False)
        self.W_res_1 = torch.nn.Parameter(
            torch.from_numpy(W_res_1).float().to(self._device), requires_grad=False)
        self.W_res_2 = torch.nn.Parameter(
            torch.from_numpy(W_res_2).float().to(self._device), requires_grad=False)
        self._w_in_flat = self.W_in[0].view(1, 1, 1, -1) if int(self.W_in.shape[0]) == 1 else None

    def update_reservoir(self, x, state1, state2):
        if x.device != self._device:
            x = x.to(self._device)
        pre_activation = torch.mm(x.unsqueeze(1), self.W_in) + torch.mm(state1, self.W_res_1) + torch.mm(state2, self.W_res_2)
        state_new = self.activation(pre_activation)
        return state_new

    def update_reservoir_from_input_term(self, input_term, state1, state2):
        if input_term.device != self._device:
            input_term = input_term.to(self._device)
        pre_activation = input_term + torch.mm(state1, self.W_res_1) + torch.mm(state2, self.W_res_2)
        state_new = self.activation(pre_activation)
        return state_new

    def ridge_regression(self, states, targets, with_bias=True):
        batch, length, dim = states.shape
        targets = targets.unsqueeze(2)
        if with_bias:
            ones = torch.ones((batch, length, 1), dtype=states.dtype, device=states.device)
            states = torch.cat([states, ones], dim=2)
            dim = dim + 1

        if (not hasattr(self, "_ridge_I_base")) or (self._ridge_I_base is None) or (int(self._ridge_I_base.shape[-1]) != int(dim)):
            self._ridge_I_base = torch.eye(dim, dtype=states.dtype, device=states.device).unsqueeze(0)
        I = self._ridge_I_base.expand(batch, -1, -1) * float(self.alpha) * float(self.alpha)

        XtX = torch.bmm(states.transpose(1, 2), states)
        XtX_plus_lambdaI = XtX + I
        Xty = torch.bmm(states.transpose(1, 2), targets)

        import os
        solver_env = (os.environ.get("RES_SAM_RIDGE_SOLVER", "") or "").strip().lower()
        solver = solver_env or "inverse"
        if solver == "solve":
            W_out = torch.linalg.solve(XtX_plus_lambdaI, Xty)
        else:
            XtX_plus_lambdaI_inv = torch.inverse(XtX_plus_lambdaI)
            W_out = torch.bmm(XtX_plus_lambdaI_inv, Xty)

        return W_out.squeeze(2)

    def forward(self, input_image):
        """Forward pass: [batch_size, height, width] -> features [batch, 2*n_reservoir(+1)]."""
        input_image = input_image.detach().to(self._device)
        batch, height, width = input_image.shape
        state = torch.zeros((batch, height, width, self.n_reservoir), dtype=torch.float32, device=self._device)
        initial_state = torch.zeros((batch, self.n_reservoir), dtype=torch.float32, device=self._device)

        state[:, 0, 0, :] = self.update_reservoir(input_image[:, 0, 0], initial_state, initial_state)
        for t in range(1, width):
            state[:, 0, t, :] = self.update_reservoir(input_image[:, 0, t], state[:, 0, t - 1, :], initial_state)
        for h in range(1, height):
            state[:, h, 0, :] = self.update_reservoir(input_image[:, h, 0], initial_state, state[:, h - 1, 0, :])
            for t in range(1, width):
                state[:, h, t, :] = self.update_reservoir(
                    input_image[:, h, t], state[:, h, t - 1, :], state[:, h - 1, t, :]
                )

        pre_states, targets = [], []
        for h in range(self.start_node[0], height):
            for t in range(self.start_node[1], width):
                new_state = torch.cat([state[:, h, t - 1, :], state[:, h - 1, t, :]], dim=1).unsqueeze(1)
                pre_states.append(new_state)
                targets.append(input_image[:, h, t].unsqueeze(1))

        pre_states = torch.cat(pre_states, dim=1)
        targets = torch.cat(targets, dim=1)
        return self.ridge_regression(pre_states, targets, with_bias=True)

    def forward_masked(self, input_image, valid_mask):
        """Fit one irregular masked region as a single 2D-ESN model."""
        input_image = input_image.detach().to(self._device)
        valid_mask = valid_mask.to(self._device).bool()

        if input_image.ndim != 3 or valid_mask.ndim != 3:
            raise ValueError("forward_masked expects [batch, height, width] tensors")
        if input_image.shape != valid_mask.shape:
            raise ValueError("input_image and valid_mask must have the same shape")

        batch, height, width = input_image.shape
        if batch != 1:
            features = [self.forward_masked(input_image[i:i + 1], valid_mask[i:i + 1]) for i in range(batch)]
            return torch.cat(features, dim=0)

        state = torch.zeros((1, height, width, self.n_reservoir), dtype=torch.float32, device=self._device)
        initial_state = torch.zeros((1, self.n_reservoir), dtype=torch.float32, device=self._device)

        for h in range(height):
            for t in range(width):
                if not bool(valid_mask[0, h, t]):
                    continue
                state1 = state[:, h, t - 1, :] if t > 0 else initial_state
                state2 = state[:, h - 1, t, :] if h > 0 else initial_state
                state[:, h, t, :] = self.update_reservoir(input_image[:, h, t], state1, state2)

        pre_states, targets = [], []
        for h in range(self.start_node[0], height):
            for t in range(self.start_node[1], width):
                if not bool(valid_mask[0, h, t]):
                    continue
                pre_states.append(
                    torch.cat([state[:, h, t - 1, :], state[:, h - 1, t, :]], dim=1).unsqueeze(1)
                )
                targets.append(input_image[:, h, t].unsqueeze(1))

        feature_dim = 2 * self.n_reservoir + 1
        if not pre_states:
            return torch.zeros((1, feature_dim), dtype=torch.float32, device=self._device)

        pre_states = torch.cat(pre_states, dim=1)
        targets = torch.cat(targets, dim=1)
        return self.ridge_regression(pre_states, targets, with_bias=True)
