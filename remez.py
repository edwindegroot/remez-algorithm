from __future__ import division
from matplotlib import pyplot as plt
import bisect

import numpy as np
import scipy


class IterationResult:
    def __init__(self, new_reference, error, approx):
        self.new_reference = new_reference
        self.error = error
        self.approx = approx


class Remez:
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
        abs_max_of_residue_function = abs_residue_function(abs_max_pos_of_residue_function)
        print(abs_max_of_residue_function)
        z_array = [interval_start]
        for i in range(1, len(reference)):
            z_array.append(self.find_root(reference[i - 1], reference[i], residue_function))
        z_array.append(interval_end)

        new_reference = []
        done = False
        for i in range(0, len(reference)):
            section_start = z_array[i]
            section_end = z_array[i + 1]
            sign = np.sign(residue_function(reference[i]))
            signed_residue_function = lambda x: sign * residue_function(x)
            interval_max_pos = self.find_max_pos(signed_residue_function, section_start, section_end)
            current_value = signed_residue_function(interval_max_pos)
            if abs_max_of_residue_function <= current_value:
                done = True
            val_at_ref_i = signed_residue_function(reference[i])
            if val_at_ref_i > current_value:
                new_reference.append(reference[i])
            else:
                new_reference.append(interval_max_pos)
        if done:
            # One of the indices of the new basis gave a higher value than the computed global max. Done with this
            # iteration.
            return new_reference
        # Otherwise, insert the position of the global max into the new reference correctly
        bisect.insort(new_reference, abs_max_pos_of_residue_function)
        index_to_remove = -1
        for i in range(0, len(new_reference) - 1):
            val_i = residue_function(new_reference[i])
            val_i_plus_one = residue_function(new_reference[i + 1])
            if np.sign(val_i) == np.sign(val_i_plus_one):
                if abs(val_i) > abs(val_i_plus_one):
                    index_to_remove = i + 1
                    break
                else:
                    index_to_remove = i
        if index_to_remove == -1:
            print('ERROR')
        del new_reference[index_to_remove]
        return new_reference

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
    function = lambda x: abs(pow(x, 9))
    basis = [
        None, # Dummy value to fill up g0
        lambda x: 1,
        lambda x: x,
        lambda x: pow(x, 2),
        lambda x: pow(x, 3)
    ]
    interval_start = -1
    interval_end = 1
    reference = np.linspace(interval_start, interval_end, len(basis))

    for i in range(0, 10):
        reference = remez.one_iteration(function, basis, interval_start, interval_end, reference)
