const projects = [
  ['01', 'optimization tools', ['Optimization', 'Python'], 'OptimizationAlgorithm', 'A code-first toolkit for studying heuristic optimizers, from PSO and GA to differential evolution, simulated annealing, and NSGA-II.', 'OptimizationAlgorithm'],
  ['02', 'power-electronics ai-ml', ['Power electronics', 'PINN'], 'PINN Modulation for SIMO Converters', 'Physics-informed learning for converter modulation, combining data-driven models with the structure and constraints of the physical system.', 'PINN_modulation_SIMO_converter'],
  ['03', 'power-electronics ai-ml', ['Power electronics', 'AI / ML'], 'KANN for LLC Voltage Gain', 'Neural-network modeling of LLC resonant converter voltage gain for faster analysis and design-space exploration.', 'KANN_LLC_Voltage_Gain'],
  ['04', 'power-electronics ai-ml', ['EV charging', 'AI / ML'], 'EV Charging Efficiency Prediction', 'A KU Leuven project exploring data-driven prediction of electric-vehicle charging efficiency.', 'EV_charging_efficiency_prediction_KUL'],
  ['05', 'power-electronics ai-ml', ['Battery systems', 'Deep learning'], 'Battery Temperature & Charger Efficiency', 'An attention-augmented multi-output neural network for jointly estimating battery temperature and charger efficiency.', 'Attention-Augmented-Multi-Output-Neural-Network-for-Battery-Temperature-and-Charger-Efficiency'],
  ['06', 'ai-ml tools', ['Reinforcement learning', 'Python'], 'Reinforcement Learning', 'Implementations and learning examples spanning value-based, policy-based, and continuous-control methods.', 'Reinforcement-Learning'],
  ['07', 'ai-ml tools', ['Reinforcement learning', 'TensorFlow'], 'RL with TensorFlow', 'TensorFlow-based reinforcement-learning experiments for connecting algorithm concepts to executable models.', 'Reinforcement-learning-with-tensorflow'],
  ['08', 'ai-ml tools', ['Machine learning', 'Python'], 'Python for Machine Learning', 'Practical notebooks and examples for building a foundation in Python-based machine-learning workflows.', 'Python-for-machine-learning'],
  ['09', 'optimization ai-ml tools', ['Optimization', 'Machine learning'], 'Hyperparameter Tuning with Python', 'Examples of systematic model tuning and search strategies for machine-learning experiments.', 'Hyperparameter-Tuning-with-Python'],
  ['10', 'power-electronics ai-ml tools', ['Magnetics', 'AI / ML'], 'MagNet', 'Work around data-driven modeling of magnetic materials and core-loss behavior for power-electronics design.', 'MagNet']
];

const catalog = document.querySelector('[data-catalog]');
catalog.innerHTML = projects.map(([number, topics, labels, title, description, repository]) => `
  <article class="library-card" data-list-item data-topic="${topics}" data-search="${title} ${description} ${labels.join(' ')}">
    <span class="card-number">PROJECT / ${number}</span>
    <div class="archive-meta">${labels.map((label) => `<span>${label}</span>`).join('')}</div>
    <h2>${title}</h2><p>${description}</p>
    <a class="card-link" href="https://github.com/FanghaoT/${repository}" target="_blank" rel="noreferrer">View repository →</a>
  </article>`).join('');
