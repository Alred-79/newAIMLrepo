# Contributing Guide

Thank you all for helping move this project forward fr, This document explains how to:

1. Set up your local environment  
2. Create well-formed branches and PRs  
3. Run common commands  

---

## 1. Prerequisites (running latest)

- **Git** (v2.20+)  
- **Node.js** (v18+) for JS repos  
- **Python** (v3.10+) for `ai-ml`  
- **Docker** (optional, for `ai-ml`)  
- **Terraform** (v1.5+) for `infra`  

---

## 2. Clone & Bootstrap

```bash
# 1. Clone your fork or repo
git clone git@github.com:Rutgers-ClinEQ/<repo-name>.git
cd <repo-name>

# 2. Install dependencies as stated:
#    frontend & common: npm
#    backend: as needed
#    ai-ml: pip (after adding requirements.txt)
#    infra: none locally beyond Terraform
