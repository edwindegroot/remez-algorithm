from __future__ import division

import math

import numpy as np
import scipy

class Remez:
    def one_iteration(self,
                      function,
                      basis,
                      interval_start,
                      interval_end,
                      reference,
                      delta):
        n = len(basis)
        matrix = []
        right_hand_side = []
        for i in range(0, n):
            right_hand_side.append(self.get_right_coordinate(function, reference, i))
            row = []
            matrix.append(row)
            for j in range(0, n):
                matrix[i].append(self.get_left_coordinate(basis, reference, i, j))
        np_array = np.array(matrix)
        answer = np.linalg.solve(np_array, right_hand_side)
        z_array = []
        z_array.append(interval_start)
        for i in range(1, n):
            z_array.append(self.find_root(reference[i - 1], reference[i], function, answer, basis))
        z_array.append(interval_end)
        signs = []
        for ref in reference:
            signs.append(np.sign(self.difference_function(ref, function, answer, basis)))
        new_reference = []
        for i in range(0, n):
            current_interval_start = z_array[i]
            current_interval_end = z_array[i + 1]
            current_function = lambda x: signs[i] * self.difference_function(x, function, answer, basis)
            new_reference.append(self.find_max(current_function, current_interval_start, current_interval_end))
        return ''

    def find_max(self, function, start_incl, end_incl):
        current_max = None
        current_max_pos = None
        points = np.linspace(start_incl, end_incl, 100)
        for point in points:
            current_val = function(point)
            if current_max is None:
                current_max = current_val
            if current_val > current_max:
                current_max = current_val
                current_max_pos = point
        return current_max_pos

    def find_root(self, start, end, function, answer, basis):
        fn = lambda x: self.difference_function(x, function, answer, basis)
        return scipy.optimize.brentq(fn, start, end)

    def difference_function(self, x, function, coordinates, basis):
        sum_val = self.sum_function(x, coordinates, basis)
        function_val = function(x)
        return function_val - sum_val

    def sum_function(self, x, coordinates, basis):
        result = 0
        n = len(coordinates)
        for i in range(0, n):
            result += coordinates[i] * basis[i](x)
        return result

    def get_left_coordinate(self, basis, reference, i, j):
        current_function = basis[j]
        return current_function(reference[i]) - pow(-1, i + 1) * current_function(reference[0])

    def get_right_coordinate(self, function, reference, i):
        return function(reference[i]) - pow(-1, i + 1) * function(reference[0])


if __name__ == '__main__':
    remez = Remez()
    function = lambda x: abs(x)
    basis = [
        lambda x: pow(x, 2),
        lambda x: x,
        lambda x: 1
    ]
    interval_start = -3
    interval_end = 3
    reference = np.linspace(interval_start, interval_end, 4)
    delta = 0.1
    remez.one_iteration(function, basis, interval_start, interval_end, reference, delta)



