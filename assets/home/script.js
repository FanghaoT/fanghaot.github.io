const root = document.documentElement;
const themeButton = document.querySelector('.theme-toggle');
const navButton = document.querySelector('.nav-toggle');
const nav = document.querySelector('.main-nav');

const storedTheme = localStorage.getItem('fanghao-theme');
const preferredTheme = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
root.dataset.theme = storedTheme || preferredTheme;

themeButton?.addEventListener('click', () => {
  const nextTheme = root.dataset.theme === 'dark' ? 'light' : 'dark';
  root.dataset.theme = nextTheme;
  localStorage.setItem('fanghao-theme', nextTheme);
});

navButton?.addEventListener('click', () => {
  const expanded = navButton.getAttribute('aria-expanded') === 'true';
  navButton.setAttribute('aria-expanded', String(!expanded));
  nav?.classList.toggle('open', !expanded);
});

nav?.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    nav.classList.remove('open');
    navButton?.setAttribute('aria-expanded', 'false');
  });
});

const observer = new IntersectionObserver(
  (entries) => entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      observer.unobserve(entry.target);
    }
  }),
  { threshold: 0.12 }
);

document.querySelectorAll('.reveal').forEach((element) => observer.observe(element));
