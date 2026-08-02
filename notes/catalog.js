const notes = [
  ['2026-08-01', '01 AUG 2026', 'optimization python', ['Optimization', 'Python'], 'Heuristic Optimization Algorithms — A Code-First Learning Series', 'From-PSO-to-NSGA-II-Building-an-Optimization-Algorithm-Toolkit'],
  ['2024-04-05', '05 APR 2024', 'reinforcement-learning', ['Reinforcement learning'], 'Reinforcement Learning — Summary and Outlook', 'The-Future-of-Reinforcement-Learning-Summary-and-Outlook'],
  ['2024-03-29', '29 MAR 2024', 'reinforcement-learning', ['Reinforcement learning', 'Exploration'], 'Advanced Topics — Dyna-Q and Curiosity-Driven Learning', 'Advanced-Topics-Dyna-Q-and-Curiosity-Driven-Learning'],
  ['2024-03-22', '22 MAR 2024', 'reinforcement-learning', ['Reinforcement learning'], 'PPO — Proximal Policy Optimization', 'PPO-Proximal-Policy-Optimization'],
  ['2024-03-15', '15 MAR 2024', 'reinforcement-learning', ['Reinforcement learning'], 'A3C — Asynchronous Advantage Actor-Critic', 'A3C-Asynchronous-Advantage-Actor-Critic'],
  ['2024-03-08', '08 MAR 2024', 'reinforcement-learning', ['Reinforcement learning', 'Control'], 'DDPG — Deep RL for Continuous Control', 'DDPG-Deep-RL-for-Continuous-Control'],
  ['2024-03-01', '01 MAR 2024', 'reinforcement-learning', ['Reinforcement learning'], 'Actor-Critic — Best of Both Worlds', 'Actor-Critic-Best-of-Both-Worlds'],
  ['2024-02-16', '16 FEB 2024', 'reinforcement-learning python', ['Reinforcement learning', 'Python'], 'OpenAI Gym — Your RL Playground', 'OpenAI-Gym-Your-RL-Playground'],
  ['2024-02-09', '09 FEB 2024', 'reinforcement-learning', ['Reinforcement learning'], 'DQN Improvements — Double, Dueling, and Prioritized Experience', 'DQN-Improvements-Double-Dueling-and-Prioritized-Experience'],
  ['2024-02-02', '02 FEB 2024', 'reinforcement-learning', ['Reinforcement learning'], 'Enter the Deep — Deep Q-Networks (DQN)', 'Enter-the-Deep-Deep-Q-Networks-DQN'],
  ['2024-01-26', '26 JAN 2024', 'reinforcement-learning', ['Reinforcement learning'], 'SARSA Lambda — Adding Memory with Eligibility Traces', 'SARSA-Lambda-Adding-Memory-with-Eligibility-Traces'],
  ['2024-01-19', '19 JAN 2024', 'reinforcement-learning', ['Reinforcement learning'], 'SARSA — On-Policy Learning in Action', 'SARSA-On-Policy-Learning-in-Action'],
  ['2024-01-12', '12 JAN 2024', 'reinforcement-learning', ['Reinforcement learning'], 'Tabular Q-Learning — Navigating the Maze', 'Tabular-Q-Learning-Navigating-the-Maze'],
  ['2024-01-05', '05 JAN 2024', 'reinforcement-learning', ['Reinforcement learning'], 'Introduction to Reinforcement Learning — The Treasure Hunt', 'Introduction-to-Reinforcement-Learning-The-Treasure-Hunt'],
  ['2023-07-21', '21 JUL 2023', 'python', ['Python'], 'Compare os.path and pathlib on Calling Files', 'Compare-ospath-and-pathlib-on-calling-files'],
  ['2023-07-14', '14 JUL 2023', 'python', ['Python', 'Machine learning'], 'ML Tutorial 01 — Installing Anaconda and VS Code for AI Work', 'Introduction-to-Installing-Anaconda-and-VS-Code'],
  ['2023-07-07', '07 JUL 2023', 'python', ['Python'], 'Common Mistakes in Python Logic Operations', 'Logic-Operation-in-Python'],
  ['2023-06-26', '26 JUN 2023', 'power-electronics', ['Power electronics', 'PSIM'], 'A Demonstration of the Script Function in PSIM', 'Scripts-in-PSIM'],
  ['2023-06-09', '09 JUN 2023', 'general', ['General'], 'Some Starting Words', 'Some-Starting-Words']
];

const catalog = document.querySelector('[data-catalog]');
catalog.innerHTML = notes.map(([isoDate, date, topics, labels, title, slug]) => `
  <a class="archive-item" href="/posts/${slug}/" data-list-item data-topic="${topics}" data-search="${title} ${labels.join(' ')}">
    <time class="archive-date" datetime="${isoDate}">${date}</time>
    <div><div class="archive-meta">${labels.map((label) => `<span>${label}</span>`).join('')}</div><h2>${title}</h2></div><b>→</b>
  </a>`).join('');
