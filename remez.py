from __future__ import division
from matplotlib import pyplot as plt

import numpy as np
import scipy


class RemezResult:
    def __init__(self, error, approx):
        self.error = error
        self.approx = approx


class IterationResult:
    def __init__(self, new_reference, error, approx):
        self.new_reference = new_reference
        self.error = error
        self.approx = approx


class Remez:
    def calculate_approximation_monomial_basis(
            self,
            function,
            polynomial_degree,
            interval_start,
            interval_end
    ):
        basis = [None]
        for i in range(0, polynomial_degree + 1):
            basis.append(lambda x, i=i: pow(x, i))
        return self.calculate_approximation(function, basis, interval_start, interval_end)

    def calculate_approximation(self,
                                function,
                                basis,
                                interval_start,
                                interval_end):
        reference = np.linspace(interval_start, interval_end, len(basis))
        current_error = -1
        result = IterationResult(reference, None, None)
        for i in range(0, 10):
            result = remez.one_iteration(function, basis, interval_start, interval_end, result.new_reference)
            new_error = result.error
            if current_error == -1:
                current_error = new_error
            else:
                if new_error >= current_error or current_error - new_error < 0.000000001:
                    break
                else:
                    current_error = new_error
        remez_result = RemezResult(result.error, result.approx)
        return remez_result

    def one_iteration(self,
                      function,
                      basis,
                      interval_start,
                      interval_end,
                      reference):
        solution = self.compute_approx(function, reference, basis)
        approx_function = lambda x: self.approx_function(x, solution, basis)
        residue_function = lambda x: self.residue_function(x, function, approx_function)
        abs_residue_function = lambda x: abs(residue_function(x))
        abs_max_pos_of_residue_function = self.find_max_pos(abs_residue_function, interval_start, interval_end)
        max_error = abs_residue_function(abs_max_pos_of_residue_function)
        z_array = self.get_zeroes_array(residue_function, reference, interval_start, interval_end)
        new_reference = self.calculate_new_reference(z_array, residue_function, reference)
        result = IterationResult(new_reference, max_error, approx_function)
        return result

    def calculate_new_reference(self, z_array, residue_function, reference):
        new_reference = []
        for i in range(0, len(reference)):
            signed_residue_function = lambda x: np.sign(residue_function(reference[i])) * residue_function(x)
            interval_max_pos = self.find_max_pos(signed_residue_function, z_array[i], z_array[i + 1])
            current_value = signed_residue_function(interval_max_pos)
            val_at_ref_i = signed_residue_function(reference[i])
            if val_at_ref_i > current_value:
                new_reference.append(reference[i])
            else:
                new_reference.append(interval_max_pos)
        return new_reference

    def get_zeroes_array(self, residue_function, reference, interval_start, interval_end):
        z_array = [interval_start]
        for i in range(1, len(reference)):
            z_array.append(self.find_root(reference[i - 1], reference[i], residue_function))
        z_array.append(interval_end)
        return z_array

    def approx_function(self, x, solution, basis):
        result = 0
        n = len(solution)
        for i in range(1, n + 1):
            result += solution[i - 1] * basis[i](x)
        return result

    def plot(self, function, approx, residue, reference, interval_start, interval_end):
        x = np.linspace(interval_start, interval_end, 1000)
        plt.plot(x, function(x), 'red')
        plt.plot(x, residue(x), 'blue')
        plt.plot(x, approx(x), 'green')
        for ref in reference:
            x_values = [ref]
            y_values = [0]
            plt.plot(x_values, y_values, 'bo')
        plt.show()

    def compute_approx(self, function, reference, basis):
        n = len(reference) - 1
        matrix = []
        right_hand_side = []
        for i in range(1, n + 1):
            right_hand_side.append(self.get_right_coordinate(function, reference, i))
            row = []
            matrix.append(row)
            for j in range(1, n + 1):
                matrix[i - 1].append(self.get_left_coordinate(basis, reference, i, j))
        np_array = np.array(matrix)
        return np.linalg.solve(np_array, right_hand_side)

    def get_left_coordinate(self, basis, reference, i, j):
        current_function = basis[j]
        return current_function(reference[i]) - pow(-1, i) * current_function(reference[0])

    def get_right_coordinate(self, function, reference, i):
        return function(reference[i]) - pow(-1, i) * function(reference[0])

    def find_max_pos(self, function, start_incl, end_incl):
        current_max = None
        current_max_pos = None
        points = np.linspace(start_incl, end_incl, 10000)
        for point in points:
            current_val = function(point)
            if current_max is None or current_val > current_max:
                current_max = current_val
                current_max_pos = point
        return current_max_pos

    def find_root(self, start, end, function):
        return scipy.optimize.brentq(function, start, end)

    def residue_function(self, x, function, approx):
        sum_val = approx(x)
        function_val = function(x)
        return function_val - sum_val


if __name__ == '__main__':
    remez = Remez()
    function = lambda x: pow(x, 15)
    interval_start = -2
    interval_end = 2
    result = remez.calculate_approximation_monomial_basis(function, 4, interval_start, interval_end)

    x = np.linspace(interval_start, interval_end, 100)
    plt.plot(x, result.approx(x), 'red')
    plt.plot(x, function(x), 'blue')
    plt.show()
