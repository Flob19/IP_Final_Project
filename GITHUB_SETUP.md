# GitHub Setup Instructions

## Lokalt Git-repository är redan skapat! ✅

## Nästa steg: Skapa GitHub Repository

### Alternativ 1: Via GitHub Web Interface (Enklast)

1. **Skapa nytt repository på GitHub:**
   - Gå till: https://github.com/new
   - Repository name: `IP_Final_Project` (eller valfritt namn)
   - Description: "Super-Resolution GAN project with SRCNN, SRGAN, and Attentive ESRGAN"
   - Välj Public eller Private
   - **VIKTIGT**: Klicka INTE på "Add a README file" (vi har redan en)
   - Klicka "Create repository"

2. **Koppla lokalt repo till GitHub:**
   ```bash
   cd /Users/felixfloberg/Downloads/IP_Final_Project
   git remote add origin https://github.com/DITT-ANVÄNDARNAMN/REPO-NAMN.git
   ```
   (Ersätt DITT-ANVÄNDARNAMN och REPO-NAMN med dina värden)

3. **Pusha till GitHub:**
   ```bash
   git push -u origin main
   ```

### Alternativ 2: Via GitHub CLI (om installerat)

Om du har GitHub CLI (`gh`) installerat:

```bash
cd /Users/felixfloberg/Downloads/IP_Final_Project
gh repo create IP_Final_Project --public --source=. --remote=origin --push
```

### Verifiera att allt fungerade

Efter att ha pushat, kontrollera:
```bash
git remote -v
git log --oneline
```

Du bör nu kunna se ditt repository på GitHub!

## Uppdatera README med rätt URL

Efter att ha skapat GitHub-repositoryt, uppdatera README.md och ersätt:
- `<repository-url>` med din faktiska GitHub URL

