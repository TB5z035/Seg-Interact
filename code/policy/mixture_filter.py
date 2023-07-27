"""
This file intends to fit the distribution of a data array with mixture of two distributions
Supported distributions:
    - Gamma
    - Beta
"""

# %%
import copy
import os
from typing import Any, Callable, Tuple, Optional

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scipy.special as sp

########################################################################################################################
## Possibility Distribution Functions


class PDFunction:

    def __init__(self, *args) -> None:
        self.params = [*args]

    def update(self, *args):
        self.params = [*args]

    def max(self):
        raise NotImplementedError()

    def __call__(self, t):
        raise NotImplementedError

    def em_step(self, arr, prob):
        raise NotImplementedError

    def sample(self) -> float:
        raise NotImplementedError


class GammaDistribution(PDFunction):
    """
    $f(x) = \frac{b^a}{\Gamma(a)} e^{-bx} x^{a-1}$
    """

    def __call__(self, t):
        a, b = self.params
        return b**a / (sp.gamma(a)) * np.e**(-b * t) * t**(a - 1)

    def max(self):
        return (self.params[0] - 1) / (self.params[1])

    def em_step(self, arr, prob):
        target = np.log((prob * arr).sum() / prob.sum()) - (prob * np.log(arr)).sum() / prob.sum()
        coef = prob.sum() / np.maximum((prob * arr).sum(), 1e-8)
        func = lambda x: np.log(x + 1e-5) - sp.digamma(x + 1e-5) - target
        jac = lambda x: 1 / x - sp.gamma(x)
        root = opt.root(func, self.params[0], jac=jac)
        # self.update(root.x[0], root.x[0] * coef)
        if root.x[0] == np.nan:
            raise ValueError("Nan value detected")
        return GammaDistribution(root.x[0], root.x[0] * coef)

    def sample(self) -> float:
        return np.random.gamma(self.params[0], 1 / self.params[1])

    @classmethod
    def gen_params(cls):
        while True:
            a1 = np.random.uniform(1.1, 20)
            b1 = np.random.uniform(5, 30)
            a2 = np.random.uniform(10, 20)
            b2 = np.random.uniform(1.01, 20)
            if a1 < a2 and b1 > b2:
                return [a1, b1], [a2, b2]


class BetaDistribution(PDFunction):
    """
    $f(x) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} x^{a-1} (1-x)^{b-1}$
    """

    def __call__(self, t):
        a, b = self.params
        return sp.gamma(a + b) / sp.gamma(a) / sp.gamma(b) * t**(a - 1) * (1 - t)**(b - 1)

    def max(self):
        return (self.params[0] - 1) / (self.params[0] + self.params[1] - 2)

    def em_step(self, arr, prob):
        target_a = -(prob * np.log(arr)).sum() / prob.sum()
        target_b = -(prob * np.log(1 - arr)).sum() / prob.sum()

        def func(x):
            polys = sp.polygamma(0, [x[0] + x[1], x[0], x[1]])
            return [polys[0] - polys[1] - target_a, polys[0] - polys[2] - target_b]

        def jac(x):
            polys = sp.polygamma(1, [x[0] + x[1], x[0], x[1]])
            return [[polys[0] - polys[1], polys[0]], [polys[0], polys[0] - polys[2]]]

        root = opt.root(func, self.params, jac=jac)
        # self.update(root.x[0], root.x[1])
        if root.x[0] == np.nan or root.x[1] == np.nan:
            raise ValueError("Nan value detected")
        return BetaDistribution(root.x[0], root.x[1])

    def sample(self) -> float:
        return np.random.beta(self.params[0], self.params[1])

    @classmethod
    def gen_params(cls):
        while True:
            a1 = np.random.uniform(2, 20)
            b1 = np.random.uniform(20, 2)
            a2 = np.random.uniform(2, 20)
            b2 = np.random.uniform(20, 2)
            if a1 < a2 and b1 > b2:
                return [a1, b1], [a2, b2]


class MixtureDistribution:

    def __init__(self, distributions: list[PDFunction], weight=0.5) -> None:
        assert len(distributions) == 2, f"Only support 2 distributions for now: {distributions}"
        self.dist_a: PDFunction = distributions[0]
        self.dist_b: PDFunction = distributions[1]
        self.weight = weight
        self.call_a = lambda x: self.weight * self.dist_a(x)
        self.call_b = lambda x: (1 - self.weight) * self.dist_b(x)

    def __call__(self, arr) -> Any:
        return self.weight * self.dist_a(arr) + (1 - self.weight) * self.dist_b(arr)

    def __str__(self) -> str:
        return (f'Distribution 1 params: {self.dist_a.params}\t') + (
            f'Distribution 2 params: {self.dist_b.params}\t') + (f'Weight: {self.weight}')

    def em_step(self, data_arr) -> None:
        pdf_a = self.dist_a(data_arr)
        pdf_b = self.dist_b(data_arr)
        pdf_sum = self.weight * pdf_a + (1 - self.weight) * pdf_b
        prob_a = self.weight * pdf_a / pdf_sum
        prob_b = (1 - self.weight) * pdf_b / pdf_sum
        return MixtureDistribution(
            [self.dist_a.em_step(data_arr, prob_a),
             self.dist_b.em_step(data_arr, prob_b)],
            prob_a.sum() / len(data_arr),
        )

    def test(self, data_arr, init=0.01) -> None:
        root = opt.root(self, data_arr.mean()).x[0]
        return data_arr < root

    def sample(self) -> float:
        if np.random.rand() < self.weight:
            return self.dist_a.sample()
        else:
            return self.dist_b.sample()

    def vis(self):
        visualize_callable(self, (0, 10), color='g')
        visualize_callable(self.call_a, (0, 10), color='r')
        visualize_callable(self.call_b, (0, 10), color='b')
        plt.show()


def visualize_callable(func: Callable, boundary: Tuple[float, float] = None, nstep=1000, color='green'):
    x = np.linspace(*boundary, nstep)
    y = func(x)
    plt.plot(x, y, color=color, alpha=0.75)


def visualize_distribution(data_arr: np.ndarray, bins=500, boundary=None):
    plt.hist(data_arr, bins=bins, density=True, alpha=0.5, color='g', range=boundary)


def visualize_fitting(calc: MixtureDistribution, data_arr: np.ndarray, save_dir=None, step=None):
    """
    Visualize the fitting results, save to `save_dir` if specified
    """
    boundaries = (data_arr.min(), data_arr.max())

    plt.title("Fitting Results" + (f' (Step #{step})' if step is not None else ''))
    visualize_distribution(data_arr)
    visualize_callable(calc, boundaries, color='green')
    visualize_callable(calc.call_a, boundaries, color='red')
    visualize_callable(calc.call_b, boundaries, color='blue')

    ## Save
    save_path = os.path.join(save_dir, f'{step}.png') if save_dir is not None else None
    if save_path is None:
        plt.show()
        plt.cla()
    else:
        try:
            os.remove(save_path)
        except:
            pass
        print("Saving...")
        plt.savefig(save_path)
        plt.cla()


def distribution_error(func: Callable, data_arr: np.ndarray, steps=50000):
    """
    Calculate the difference between the fitted probability distribution and the real distribution
    Note: this is not the proper way to calculate the error of fitting
    """
    y = np.histogram(data_arr, bins=steps, density=True)[0]
    x = np.linspace(data_arr.min(), data_arr.max(), steps)
    z = func(x)
    return np.abs(y - z).mean()


def fit(arr: np.ndarray,
        distribution: MixtureDistribution,
        step=30,
        use_optimal=True,
        quiet=True,
        visualize=False,
        save_dir=None) -> Optional[MixtureDistribution]:
    """
    Fit the mixture distribution `distribution` to the data array `arr`
    """
    results = [distribution]
    if not quiet:
        print("Initial", results[-1], f"Error: {distribution_error(results[-1], arr)}")
    try:
        for i in range(step):
            results.append(results[-1].em_step(arr))
            if not quiet:
                print(f"Step #{i}", results[-1], f"Error: {distribution_error(results[-1], arr)}")
        if not quiet:
            print(f"Final", results[-1], f"Error: {distribution_error(results[-1], arr)}")
        if visualize:
            visualize_fitting(results[-1], arr, save_dir, i)
        if use_optimal:
            return results[np.argmin([distribution_error(d, arr) for d in results])]
        else:
            return results[-1]
    except ValueError:
        return None


def mixture_filter(data_arr: np.ndarray, type: str, visualize=False, save_dir=None, step=30, quiet=True, num_trial=50):
    """
    Filter the data array `data_arr` with the mixture distribution `type`
    """
    Dist = GammaDistribution if type == 'gamma' else BetaDistribution
    fitted = []
    r = tqdm(range(num_trial)) if quiet else range(num_trial)
    for idx in r:
        if not quiet:
            print(f'#################################Fitting #{idx}###################################')
        init_params = Dist.gen_params()
        d = MixtureDistribution([Dist(*init_params[0]), Dist(*init_params[1])])
        fitted.append(fit(data_arr, d, step=step, quiet=quiet, visualize=visualize, save_dir=save_dir))
    errors = [distribution_error(d, data_arr) if d is not None else float('inf') for d in fitted]
    visualize_fitting(fitted[np.argmin(errors)], data_arr, save_dir, 'final')
    return fitted[np.argmin(errors)].test(data_arr), fitted[np.argmin(errors)], np.min(errors)


def test_gamma():
    gamma_1 = GammaDistribution(2, 3)
    gamma_2 = GammaDistribution(3, 3)
    mix = MixtureDistribution([gamma_1, gamma_2], 0.388)
    samples = np.array([mix.sample() for _ in range(100000)])
    visualize_distribution(samples)
    plt.show()

    results = mixture_filter(samples, 'gamma', step=50, num_trial=30)

    visualize_fitting(results[1], samples, step='final')
    visualize_distribution(samples[results[0]], boundary=(samples.min(), samples.max()))
    plt.show()


def test_beta():
    beta_1 = BetaDistribution(1, 3)
    beta_2 = BetaDistribution(3, 0.5)
    mix = MixtureDistribution([beta_1, beta_2], 0.388)
    samples = np.array([mix.sample() for _ in range(100000)])
    visualize_distribution(samples)
    plt.show()

    results = mixture_filter(samples, 'beta', step=50, num_trial=30)

    visualize_fitting(results[1], samples, step='final')
    visualize_distribution(samples[results[0]], boundary=(samples.min(), samples.max()))
    plt.show()


if __name__ == '__main__':
    test_gamma()
    test_beta()
# %%
