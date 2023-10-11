import unittest
import numpy as np
from remez import Remez
from matplotlib import pyplot as plt


class Testing(unittest.TestCase):
    def test_basic_approximation(self):
        function = lambda x: pow(x, 15)
        interval_start = -2
        interval_end = 2
        rem = Remez()
        result = rem.calculate_approximation_monomial_basis(
            function, 4, interval_start, interval_end
        )

        x = np.linspace(interval_start, interval_end, 100)
        plt.plot(x, result.approx(x), 'red')
        plt.plot(x, function(x), 'blue')
        plt.show()


if __name__ == '__main__':
    test = Testing()
