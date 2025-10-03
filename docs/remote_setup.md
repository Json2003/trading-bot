# Remote Authentication Setup

To authenticate pushes to the target repository from a local checkout, update the `origin` remote to use a Personal Access Token (PAT):

```bash
git remote set-url origin https://x-access-token:${PAT_DATA_PUSH}@github.com/OWNER/REPO.git
```

Replace `PAT_DATA_PUSH` with an environment variable or inline token value that has the necessary permissions for the repository.
