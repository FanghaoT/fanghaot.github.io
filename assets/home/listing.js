const filterButtons = [...document.querySelectorAll('[data-filter]')];
const listItems = [...document.querySelectorAll('[data-list-item]')];
const searchInput = document.querySelector('[data-list-search]');
const resultCount = document.querySelector('[data-result-count]');
const emptyState = document.querySelector('[data-empty-state]');

let activeFilter = 'all';

function applyFilters() {
  const query = searchInput?.value.trim().toLowerCase() || '';
  let visibleCount = 0;

  listItems.forEach((item) => {
    const topics = (item.dataset.topic || '').split(' ');
    const content = (item.dataset.search || item.textContent || '').toLowerCase();
    const matchesTopic = activeFilter === 'all' || topics.includes(activeFilter);
    const matchesQuery = !query || content.includes(query);
    const visible = matchesTopic && matchesQuery;
    item.hidden = !visible;
    if (visible) visibleCount += 1;
  });

  if (resultCount) resultCount.textContent = String(visibleCount);
  emptyState?.classList.toggle('visible', visibleCount === 0);
}

filterButtons.forEach((button) => {
  button.addEventListener('click', () => {
    activeFilter = button.dataset.filter || 'all';
    filterButtons.forEach((candidate) => candidate.classList.toggle('active', candidate === button));
    applyFilters();
  });
});

searchInput?.addEventListener('input', applyFilters);

const requestedTopic = new URLSearchParams(window.location.search).get('topic');
const requestedButton = filterButtons.find((button) => button.dataset.filter === requestedTopic);
if (requestedButton) requestedButton.click();
else applyFilters();
