# ai-ml

Data cleaning, model fine-tuning, and active‐learning pipeline.


## Structure

- \`data/\`
  - \`raw/\`: original data dumps  
  - \`cleaned/\`: de-identified and cleaned data  
- \`preprocessing/\`: data ingestion & cleaning scripts  
- \`training/\`: HuggingFace scripts & Slurm job definitions  
- \`al_pipeline/\`: active-learning logic (low-confidence detection, SME loop)  
- \`tests/\`: simple tests to validate container and scripts  

## Docker

Build the Docker image:

```bash
docker build -t bias-ai/ai-ml:latest .