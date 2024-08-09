import numpy as np


def generated_function_D4kDx2Dy2(logtheta, logsigma, x, y):
    exp1 = np.exp(1)

    term1 = (
            6 * exp1 ** (2 * logtheta + logsigma) * (2 * exp1 ** logsigma + 1) /
            (
                    np.pi * (
                    1 - (exp1 ** logtheta * x * y + exp1 ** logsigma) ** 2 /
                    ((exp1 ** logtheta * x ** 2 + exp1 ** logsigma + 1) *
                     (exp1 ** logtheta * y ** 2 + exp1 ** logsigma + 1))
            ) ** (5 / 2) *
                    (exp1 ** logtheta * x ** 2 + exp1 ** logsigma + 1) ** (5 / 2) *
                    (exp1 ** logtheta * y ** 2 + exp1 ** logsigma + 1) ** (5 / 2)
            )
    )

    term2 = (
            -15 * exp1 ** (2 * logtheta) * (2 * exp1 ** logsigma + 1) *
            (
                    (2 * exp1 ** logtheta * x * (exp1 ** logtheta * x * y + exp1 ** logsigma) ** 2) /
                    (
                            (exp1 ** logtheta * x ** 2 + exp1 ** logsigma + 1) ** 2 *
                            (exp1 ** logtheta * y ** 2 + exp1 ** logsigma + 1)
                    ) -
                    2 * exp1 ** logtheta * y * (exp1 ** logtheta * x * y + exp1 ** logsigma) /
                    (
                            (exp1 ** logtheta * x ** 2 + exp1 ** logsigma + 1) *
                            (exp1 ** logtheta * y ** 2 + exp1 ** logsigma + 1)
                    )
            ) * (exp1 ** logsigma * x - exp1 ** logsigma * y - y) /
            (
                    np.pi * (
                    1 - (exp1 ** logtheta * x * y + exp1 ** logsigma) ** 2 /
                    ((exp1 ** logtheta * x ** 2 + exp1 ** logsigma + 1) *
                     (exp1 ** logtheta * y ** 2 + exp1 ** logsigma + 1))
            ) ** (7 / 2) *
                    (exp1 ** logtheta * x ** 2 + exp1 ** logsigma + 1) ** (5 / 2) *
                    (exp1 ** logtheta * y ** 2 + exp1 ** logsigma + 1) ** (5 / 2)
            )
    )

    term3 = (
            -30 * exp1 ** (3 * logtheta) * x * (2 * exp1 ** logsigma + 1) *
            (exp1 ** logsigma * x - exp1 ** logsigma * y - y) /
            (
                    np.pi * (
                    1 - (exp1 ** logtheta * x * y + exp1 ** logsigma) ** 2 /
                    ((exp1 ** logtheta * x ** 2 + exp1 ** logsigma + 1) *
                     (exp1 ** logtheta * y ** 2 + exp1 ** logsigma + 1))
            ) ** (5 / 2) *
                    (exp1 ** logtheta * x ** 2 + exp1 ** logsigma + 1) ** (7 / 2) *
                    (exp1 ** logtheta * y ** 2 + exp1 ** logsigma + 1) ** (5 / 2)
            )
    )

    return term1 + term2 + term3

print(generated_function_D4kDx2Dy2(0,0,-0.5,1))