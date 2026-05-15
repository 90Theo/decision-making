# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 12:49:53 2025

@author: geots
"""

"""
Price process.
NOT TO BE CHANGED BY THE STUDENTS
"""

import numpy as np
import matplotlib.pyplot as plt
import SystemCharacteristics


def price_model(current_price, previous_price):
    """
    Price process with dependence on previous prices.
    """
    mean_price = 4
    reversion_strength = 0.12
    price_cap = 12
    price_floor = 0

    mean_reversion = reversion_strength * (mean_price - current_price)
    noise = np.random.normal(0, 0.5)

    next_price = current_price + 0.6 * (current_price - previous_price) + mean_reversion + noise

    # Special handling if price goes negative
    if next_price < 0:
        if np.random.rand() > 0.2:
            next_price = np.random.uniform(0, mean_price * 0.3)

    # Enforce bounds
    return max(min(next_price, price_cap), price_floor)
