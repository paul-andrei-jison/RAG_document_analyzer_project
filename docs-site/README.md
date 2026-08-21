# DocuMind — Landing Page

Static how-to / landing page for DocuMind. Deployed on AWS Amplify.

## What's here

A single `index.html` with no build step — just HTML, CSS, and vanilla JS. It covers:

- What DocuMind is
- Step-by-step setup instructions with copy-able terminal commands
- Tech stack
- FAQ

---

## Local preview

Open `index.html` directly in your browser. No server needed.

---

## Deployment (AWS Amplify)

This folder is configured as a monorepo sub-app in the root `amplify.yml`.

- **App root:** `docs-site/`
- **Build command:** none (static site)
- **Base directory:** `/`

Amplify serves all files in `docs-site/` as-is. Any push to `main` triggers a redeploy.
