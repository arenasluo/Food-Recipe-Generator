# GitHub Setup Guide

Follow these steps to publish your project to GitHub:

## Step 1: Initialize Git Repository

```bash
cd /home/eric/Desktop/CS7643/CLIP+Transformer
git init
```

## Step 2: Configure Git (First Time Only)

If you haven't configured Git before:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 3: Add Files to Git

```bash
# Add all files
git add .

# Or add specific files
git add Food_to_receipe.ipynb
git add README.md
git add requirements.txt
git add .gitignore
```

## Step 4: Create Initial Commit

```bash
git commit -m "Initial commit: Food to Recipe Generator with CLIP + Transformer"
```

## Step 5: Create GitHub Repository

### Option A: Using GitHub Website

1. Go to [https://github.com](https://github.com)
2. Click the "+" button in the top right corner
3. Select "New repository"
4. Fill in the details:
   - **Repository name**: `Food-Recipe-Generator` or `CLIP-Transformer`
   - **Description**: "Deep learning system that generates recipes from food images using CLIP and GPT-2"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README (we already have one)
5. Click "Create repository"

### Option B: Using GitHub CLI (if installed)

```bash
gh repo create Food-Recipe-Generator --public --source=. --remote=origin --push
```

## Step 6: Link to GitHub Repository

After creating the repository on GitHub, connect your local repo:

```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/Food-Recipe-Generator.git

# Verify the remote
git remote -v
```

## Step 7: Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

## Step 8: Handle Large Files (Optional)

If you get errors about large files (like the model checkpoint or images):

### Option 1: Use Git LFS (Large File Storage)

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "food_images/**"

# Add the .gitattributes file
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

### Option 2: Exclude Large Files

Edit `.gitignore` to exclude large files:
```
# Model files
*.pt
*.pth

# Large datasets
food_images/
*.csv
*.zip
```

Then:
```bash
git rm --cached recipe_generator_model.pt
git commit -m "Remove large model file"
```

## Step 9: Add Additional Information

### Create a LICENSE file

```bash
# For MIT License
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

git add LICENSE
git commit -m "Add MIT License"
git push
```

## Step 10: Update README

Don't forget to update the README.md with:
- Your actual GitHub username in URLs
- Your name in citations
- Any specific setup instructions for your environment

## Troubleshooting

### Authentication Issues

If you get authentication errors:

1. **Use Personal Access Token (Recommended)**:
   - Go to GitHub Settings > Developer settings > Personal access tokens
   - Generate new token with 'repo' scope
   - Use token as password when pushing

2. **Use SSH** (Alternative):
   ```bash
   # Generate SSH key
   ssh-keygen -t ed25519 -C "your.email@example.com"

   # Add to GitHub
   cat ~/.ssh/id_ed25519.pub
   # Copy and paste into GitHub Settings > SSH Keys

   # Change remote URL
   git remote set-url origin git@github.com:YOUR_USERNAME/Food-Recipe-Generator.git
   ```

### Large File Errors

If GitHub rejects files larger than 100MB:

```bash
# Find large files
find . -type f -size +50M

# Use Git LFS or exclude them
```

### Push Rejected

If push is rejected:

```bash
# Pull first
git pull origin main --rebase

# Then push
git push origin main
```

## Quick Reference Commands

```bash
# Check status
git status

# View commit history
git log --oneline

# Create new branch
git checkout -b feature-name

# Switch branches
git checkout main

# Update from GitHub
git pull

# Push changes
git add .
git commit -m "Description of changes"
git push

# View differences
git diff

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard all local changes
git reset --hard HEAD
```

## Useful .gitignore Additions

Add these to `.gitignore` if needed:

```
# Jupyter
.ipynb_checkpoints/

# Model checkpoints
*.pt
*.pth

# Data
*.csv
*.zip
food_images/

# Environment
venv/
.env

# IDE
.vscode/
.idea/
```

## After Publishing

1. Add topics/tags on GitHub: `deep-learning`, `pytorch`, `clip`, `gpt-2`, `recipe-generation`
2. Add a description to your repository
3. Create a nice banner image for README
4. Add GitHub Actions for CI/CD (optional)
5. Enable GitHub Pages for documentation (optional)
6. Star your own repository!

## Need Help?

- GitHub Docs: https://docs.github.com
- Git Cheat Sheet: https://education.github.com/git-cheat-sheet-education.pdf
- Pro Git Book: https://git-scm.com/book/en/v2

---

**Ready to publish?** Start with Step 1!
