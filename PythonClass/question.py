multi_grid = [1, 2, 3]
atrous_rates = [grid * (6 if 16 == 16 else 12) for grid in multi_grid]
print(atrous_rates)