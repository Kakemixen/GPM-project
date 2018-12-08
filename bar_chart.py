import matplotlib.pyplot as plt

plt.style.use('ggplot')

x = ['Bayesian Network1', 'Bayesian Network 2', 'Artificial Neural Network', 'Desicion Tree']
energy = [300, 200, 25.06, 187.73]
x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, energy, color='darkblue')
plt.ylabel("Time [seconds]")
plt.title("Learning time")

plt.xticks(x_pos, x)

plt.show()