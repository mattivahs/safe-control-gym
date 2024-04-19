import numpy as np
import torch.nn as nn
import torch


def bounding_box_constraints(x_range, y_range):
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Vectors 'a' corresponding to the normal of the planes
    a = np.zeros((4, 7))
    a[:, 0:2] = np.array([
        [-1, 0],  # for x >= x_min
        [1, 0],  # for x <= x_max
        [0, -1],  # for y >= y_min
        [0, 1] # for y <= y_max
    ])

    # Scalars 'b' corresponding to the shifted values
    b_list = np.array([[
        -x_min,  # for x >= x_min
        x_max,  # for x <= x_max
        -y_min,  # for y >= y_min
        y_max  # for y <= y_max
    ]]).T

    return a, b_list


class CBFHalfspace(nn.Module):
    def __init__(self, f_dyn, g_dyn, x_range, y_range, device):
        super(CBFHalfspace, self).__init__()
        self.device = device
        A, b = bounding_box_constraints(x_range, y_range)
        self.A = torch.from_numpy(A).float().to(self.device)
        self.b = torch.from_numpy(b).float().to(self.device)
        self.f = f_dyn
        self.g = g_dyn

    def h(self, x):
        # Return a single tensor instead of a list
        return torch.stack([self.b[i] - self.A[i, :] @ x for i in range(4)])

    def dhdt(self, x):
        hs = self.h(x)
        grads = torch.autograd.grad(hs, x, grad_outputs=torch.ones_like(hs), create_graph=True)
        return torch.stack([grad.squeeze().dot(self.f.squeeze()) for grad in grads])

    def LieDerivatives(self, x):
        Lf2hs = []
        LgLfhs = []
        Lfhs = self.dhdt(x)
        for Lfh in Lfhs:
            if x.grad is not None:
                x.grad.zero_()  # Reset gradients to zero
            Lfh.backward(retain_graph=True)  # Compute gradient
            Lf2hs.append(x.grad.squeeze().dot(self.f.squeeze()))
            LgLfhs.append(x.grad @ self.g)
        return Lf2hs, LgLfhs


# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
f_dyn = torch.randn(7, 1, device=device)  # Example dynamic function output
g_dyn = torch.randn(7, 2, device=device)  # Example, replace with actual dynamics
x_range = (-1, 1)
y_range = (-1, 1)
model = CBFHalfspace(f_dyn, g_dyn, x_range, y_range, device)

x = torch.randn(7, 1, device=device, requires_grad=True)
output = model.dhdt(x)
# output = model.LieDerivatives(x)
print(output)