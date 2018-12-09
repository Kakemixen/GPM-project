import matplotlib.pyplot as plt

plt.style.use('ggplot')

x = ['Bayesian Network', 'Bayesian Network Init.', 'Artificial Neural Network', 'Desicion Tree']
energy = [152, 137, 25, 2]
x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, energy, color='darkblue')
plt.ylabel("Time [seconds]")
plt.title("Learning time")

plt.xticks(x_pos, x)

plt.show()