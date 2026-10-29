# Pithos website

The English landing page is published automatically by GitHub Pages from
`main` / `docs`. This is a static site: no Node.js build or server is required.
`.nojekyll` makes Pages serve the HTML, CSS, JavaScript, fonts, and images directly.

- `index.html`: content and metadata.
- `styles.css`: shared responsive layout.
- `theme-hack.css`: terminal-inspired colors taken from the Pithos banner.
- `app.js`: installation guide, tabs, clipboard, and local progress storage.
- `assets/`: original project graphics and self-hosted fonts with license files.

Keep asset URLs relative so the site works at `/Pithos/`. When changing the public
hostname, also update the canonical URL and Open Graph URLs in `index.html`.

The existing documentation screenshots in this folder are retained for the main
project README. Model weights, binaries, credentials, and Sites hosting metadata
are not part of the website.

Local preview:

```sh
python -m http.server 8765 --directory docs
```

Open http://127.0.0.1:8765.
